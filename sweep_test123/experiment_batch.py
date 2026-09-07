from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import threading
import time

MASTER_SEED = 5726455
PLAN_NAME = "random_experiment_plan.json"
OPTIONS = dict(
    seeds=[10, 12, 15], rounds=[120, 150, 180, 210, 240],
    arm_budget=[420, 500, 600, 780, 900, 1000],
    budget=[420_000, 500_000, 600_000, 780_000, 900_000, 1_000_000],
    pool_size=[16, 20, 24, 28, 32], feedback_delay=[2, 3, 4],
    batch_size=[2, 3, 4, 5], repayment_delay=[5, 8, 11, 14],
    recycle_fraction=[0.70, 0.80, 0.85, 0.90, 1.00],
    default_recovery=[0.00, 0.10, 0.15, 0.20, 0.25],
    liquidity_reserve=[10_000, 20_000, 30_000, 40_000],
)


def utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def generate_scenarios(num_scenarios=10, master_seed=MASTER_SEED):
    """Independent RNG; unique 100-wide seed blocks; stable count prefixes."""
    if not 1 <= num_scenarios <= 10000:
        raise ValueError("num_scenarios must be between 1 and 10000")
    rng = random.Random(master_seed)
    scenarios, used_blocks, used_configs = {}, set(), set()
    for index in range(num_scenarios):
        block = rng.randrange(100, 1_000_000)
        while block in used_blocks:
            block = rng.randrange(100, 1_000_000)
        used_blocks.add(block)
        cfg = {key: rng.choice(values) for key, values in OPTIONS.items()}
        identity = tuple(cfg[key] for key in OPTIONS if key != "seeds")
        while identity in used_configs:
            cfg = {key: rng.choice(values) for key, values in OPTIONS.items()}
            identity = tuple(cfg[key] for key in OPTIONS if key != "seeds")
        used_configs.add(identity)
        scenarios[f"scenario_{index + 1}"] = dict(seed_start=100 * block + 1, **cfg)
    return scenarios


def read_plan(path):
    plan = json.loads(Path(path).read_text())
    if generate_scenarios(plan["num_scenarios"], plan["master_seed"]) != plan["scenarios"]:
        raise ValueError("Frozen scenario plan differs from deterministic generation")
    return plan


def source_hashes(root):
    paths = [root / p for p in ("main.py", "evaluate.py", "robustness.py", "metrics.py", "simulation.py")]
    for folder in ("Algorithms", "Data", "Environments", "Helpers", "models"):
        paths.extend(sorted((root / folder).glob("*.py")))
    return {str(path.relative_to(root)): sha256(path) for path in paths}


def environment():
    return dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                packages={name: importlib.metadata.version(name) for name in ("numpy", "pandas", "matplotlib")},
                thread_limits={key: os.environ.get(key) for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS")})


def runner_hashes(root):
    return {name: sha256(root / "outputs1" / name) for name in ("run_experiments.py", "experiment_batch.py")}


def validate_provenance(root, plan):
    if source_hashes(root) != plan["source_sha256"] or runner_hashes(root) != plan["runner_sha256"]:
        raise ValueError("Source or runner changed since the frozen plan was created")
    current = environment()
    if any(current[key] != plan["environment"][key] for key in ("python", "packages", "thread_limits")):
        raise ValueError("Python or dependency versions changed since the frozen plan")


def prepare_plan(root, output, count=10, seed=MASTER_SEED, replace=False):
    output.mkdir(parents=True, exist_ok=True)
    path = output / PLAN_NAME
    if path.exists() and not replace:
        plan = read_plan(path)
        if (plan["num_scenarios"], plan["master_seed"]) != (count, seed):
            raise ValueError("Existing plan has a different count/master seed; use a new output root or --replace-existing")
        return plan, path
    scenarios = generate_scenarios(count, seed)
    existing = [output / name for name in scenarios if (output / name).exists()]
    if existing and not replace:
        raise FileExistsError("Existing scenarios have no matching plan; explicit --replace-existing is required")
    for destination in existing:
        if destination.is_symlink() or destination.parent != output:
            raise ValueError(f"Unsafe replacement path: {destination}")
    # Explicit replacement is authorised by the caller; never delete elsewhere.
    for destination in existing:
        shutil.rmtree(destination)
    plan = dict(schema_version=1, generator_version=1, master_seed=seed, num_scenarios=count,
                scenarios=scenarios, candidate_values=OPTIONS, modes=[1, 2, 3, 4], quick_test=False,
                seed_block_rule="100 * unique rng.randrange(100,1000000) + 1; contiguous IDs, at most 15 per block",
                source_sha256=source_hashes(root), runner_sha256=runner_hashes(root), environment=environment(),
                created_utc=utc_now(), parity_tolerance=0.05, contextual_parity_tolerance=0.01, quota_tolerance=0.01)
    save_json(path, plan)
    snapshot = output / "reproducibility"
    snapshot.mkdir(exist_ok=True)
    for name in plan["runner_sha256"]:
        shutil.copy2(root / "outputs1" / name, snapshot / name)
    frozen = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
    (output / "requirements_frozen.txt").write_text(frozen.stdout)
    return plan, path


def artifact_inventory(destination):
    return {str(p.relative_to(destination)): dict(bytes=p.stat().st_size, sha256=sha256(p))
            for p in sorted(destination.rglob("*")) if p.is_file()
            and p.name not in {"artifact_checksums.json", "completed.json"}}


def verify_scenario(destination, cfg, verify_checksums=False):
    import numpy as np
    import pandas as pd
    from evaluate import aggregate_results, aggregate_cmdp_results, rank_results
    from robustness import aggregate_sweep_results
    checks, phases, selections = [], {}, []
    seeds = list(range(cfg["seed_start"], cfg["seed_start"] + cfg["seeds"]))
    for phase, count in (("arm", 13), ("contextual", 14), ("cmdp", 12)):
        raw = pd.read_csv(destination / phase / "episode_summaries.csv")
        assert len(raw) == count * len(seeds) and sorted(raw.seed.unique()) == seeds, (phase, "seed matrix")
        assert raw.policy.nunique() == count and not raw.duplicated(["policy", "seed"]).any()
        assert raw.groupby("policy").size().eq(len(seeds)).all()
        agg = (aggregate_cmdp_results if phase == "cmdp" else aggregate_results)(raw)
        saved = pd.read_csv(destination / phase / "aggregate_results.csv", header=[0, 1], index_col=0)
        pd.testing.assert_frame_equal(agg.sort_index(), saved.sort_index(), check_exact=False, rtol=1e-10, atol=1e-10)
        phases[phase] = raw
        reward = raw.groupby("policy").cumulative_reward.mean()
        winner, feasible, passing = reward.idxmax(), None, None
        if phase != "arm":
            tolerance = 0.01 if phase == "contextual" else 0.05
            ranking = rank_results(agg, tolerance, 0.01)
            assert ranking.policy.tolist() == pd.read_csv(destination / phase / "ranked_results.csv").policy.tolist()
            winner, feasible = ranking.iloc[0].policy, bool(ranking.iloc[0].fairness_feasible)
            selected = raw[raw.policy == winner]
            passing = int(((selected.demographic_parity_gap <= tolerance + 1e-12) & (selected.total_quota_deficit <= 0.01 + 1e-12)).sum())
        selections.append(dict(scenario=destination.name, phase=phase, seeds=len(seeds), winner=winner,
                               winner_reward=float(reward[winner]), fairness_feasible=feasible, passing_seeds=passing,
                               raw_reward_leader=reward.idxmax(), raw_reward=float(reward.max())))
        checks.append(f"{phase}: complete seed/policy matrix, recomputed mean/SD and ranking")
    for sweep, count in (("budget", 60), ("group_imbalance", 60), ("fairness_strength", 36)):
        directory = destination / "cmdp_robustness" / "sweeps" / sweep
        raw = pd.read_csv(directory / "raw_results.csv")
        assert len(raw) == count * len(seeds) and sorted(raw.seed.unique()) == seeds, (sweep, "seed matrix")
        assert not raw.duplicated(["policy", "seed", "sweep_value", "parameter_kind"]).any()
        assert raw.groupby(["policy", "sweep_value", "parameter_kind"]).size().eq(len(seeds)).all()
        pd.testing.assert_frame_equal(aggregate_sweep_results(raw), pd.read_csv(directory / "aggregate_results.csv"), check_exact=False, rtol=1e-10, atol=1e-10)
        passing = (raw.demographic_parity_gap <= 0.05 + 1e-12) & (raw.total_quota_deficit <= 0.01 + 1e-12)
        assert (passing == raw.constraint_satisfied).all()
        baseline = raw[raw.is_baseline].set_index(["policy", "seed"]).sort_index()
        standard = phases["cmdp"].set_index(["policy", "seed"]).sort_index().loc[baseline.index]
        for column in standard.select_dtypes(include="number"):
            assert np.allclose(baseline[column], standard[column], rtol=1e-10, atol=1e-10), (sweep, column)
        checks.append(f"{sweep}: full matrix, recomputed aggregates/intervals, fairness flags and all shared numeric baseline fields")
    if verify_checksums:
        inventory = json.loads((destination / "artifact_checksums.json").read_text())
        for relative, saved in inventory.items():
            if sha256(destination / relative) != saved["sha256"]:
                raise ValueError(f"Saved artifact changed: {destination / relative}")
        checks.append("Saved artifact SHA-256 checksums verified")
    return dict(passed=True, checks=checks, phase_episodes=39 * len(seeds), sweep_episodes=156 * len(seeds), selections=selections)


def batch_report(plan, output):
    import pandas as pd
    checks, selections, configs = {}, [], []
    for name, cfg in plan["scenarios"].items():
        checks[name] = verify_scenario(output / name, cfg, verify_checksums=True)
        selections.extend(checks[name]["selections"])
        configs.append(dict(scenario=name, master_seed=plan["master_seed"], **cfg))
    save_json(output / "batch_verification.json", checks)
    pd.DataFrame(selections).to_csv(output / "batch_phase_winners.csv", index=False)
    pd.DataFrame(configs).to_csv(output / "scenario_configurations.csv", index=False)
    phases = sum(c["phase_episodes"] for c in checks.values())
    sweeps = sum(c["sweep_episodes"] for c in checks.values())
    lines = [f"# Random experiment results: {len(checks)} scenarios", "",
             f"Master seed **{plan['master_seed']}**. Modes 1–4 completed with full sweep grids: **{phases:,} phase episodes**, **{sweeps:,} sweep episodes**, **{phases+sweeps:,} total executions**.", "",
             "Every scenario passed verification of seed/policy matrices, recomputed aggregates, sweep baseline consistency, fairness flags and artifact checksums. "
             "The frozen plan, manifests, runner snapshots and requirements_frozen.txt retain reproducibility information. Sweep baseline repetitions count as executions, not independent evidence. "
             "Different scenarios jointly vary multiple parameters; absolute reward totals across scenarios are not estimates of policy improvement.", "",
             "| Scenario | Seeds | Rounds | Arm budget | Borrower capital | Pool | Loans/round | Repayment delay | Recycling |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in configs:
        lines.append(f"| {row['scenario']} | {row['seeds']} | {row['rounds']} | {row['arm_budget']} | {row['budget']:,} | {row['pool_size']} | {row['batch_size']} | {row['repayment_delay']} | {row['recycle_fraction']} |")
    lines += ["", "Phase 1 selects mean reward. Phases 2/3 prefer aggregate fairness feasibility, then reward per cost and reward; with no feasible policy they select minimum violation. Selected winners need not be feasible or maximise raw reward.", "",
              "| Scenario | Phase | Selected policy | Aggregate feasible | Passing seeds | Raw-reward leader | Mean raw reward |",
              "| --- | --- | --- | --- | --- | --- | ---: |"]
    for row in selections:
        feasible = "n/a" if row["fairness_feasible"] is None else str(row["fairness_feasible"])
        passing = "n/a" if row["passing_seeds"] is None else f"{row['passing_seeds']}/{row['seeds']}"
        lines.append(f"| {row['scenario']} | {row['phase']} | {row['winner']} | {feasible} | {passing} | {row['raw_reward_leader']} | {row['raw_reward']:.3f} |")
    lines += ["", "Run python outputs1/run_experiments.py --all to reproduce or resume this batch."]
    (output / "random_experiment_report.md").write_text("\n".join(lines) + "\n")
    return dict(scenario_count=len(checks), phase_episodes=phases, sweep_episodes=sweeps, total_episodes=phases+sweeps)


def run_all(root, plan, plan_path, output, workers):
    status = dict(master_seed=plan["master_seed"], started_utc=utc_now(), state="running", workers=workers,
                  scenarios={name: "queued" for name in plan["scenarios"]})
    mutex = threading.Lock()
    save_json(output / "batch_status.json", status)

    def launch(name):
        with mutex:
            status["scenarios"][name] = "running"
            save_json(output / "batch_status.json", status)
        print(f"START {name}", flush=True)
        command = [sys.executable, str(root / "outputs1/run_experiments.py"), name, "--plan", str(plan_path), "--output-root", str(output)]
        return subprocess.run(command).returncode

    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(launch, name): name for name in plan["scenarios"]}
        for future in as_completed(pending):
            name, code = pending[future], future.result()
            with mutex:
                status["scenarios"][name] = "complete" if code == 0 else "failed"
                save_json(output / "batch_status.json", status)
            print(f"DONE {name}: {status['scenarios'][name]}", flush=True)
    if any(value == "failed" for value in status["scenarios"].values()):
        status["state"] = "failed"
        save_json(output / "batch_status.json", status)
        raise SystemExit("Some scenarios failed; inspect terminal output and rerun --all to resume")
    totals = batch_report(plan, output)
    status.update(state="complete", completed_utc=utc_now(), **totals)
    save_json(output / "batch_status.json", status)
    print(f"BATCH COMPLETE: {totals}", flush=True)


def main(root, default_output, run):
    parser = argparse.ArgumentParser(description="Reproducible random full Modes 1--4")
    parser.add_argument("scenario", nargs="?")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--replace-existing", action="store_true")
    parser.add_argument("--master-seed", type=int, default=MASTER_SEED)
    parser.add_argument("--num-scenarios", type=int, default=10)
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=3)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output-root", type=Path, default=default_output)
    cli = parser.parse_args()
    if not (cli.scenario or cli.all or cli.plan_only or cli.verify_only):
        parser.error("Choose a scenario, --all, --plan-only or --verify-only")
    output = cli.output_root.resolve()
    if cli.plan:
        if cli.replace_existing:
            parser.error("--replace-existing cannot be combined with --plan")
        plan_path = cli.plan.resolve()
        plan = read_plan(plan_path)
    else:
        plan, plan_path = prepare_plan(root, output, cli.num_scenarios, cli.master_seed, cli.replace_existing)
    if cli.plan_only:
        print(json.dumps(dict(plan=str(plan_path), master_seed=plan["master_seed"], scenarios=plan["scenarios"]), indent=2))
    elif cli.verify_only:
        validate_provenance(root, plan)
        print(batch_report(plan, output))
    elif cli.all:
        validate_provenance(root, plan)
        run_all(root, plan, plan_path, output, cli.workers)
    elif cli.scenario in plan["scenarios"]:
        run(cli.scenario, plan["scenarios"][cli.scenario], output, plan, plan_path)
    else:
        parser.error(f"Unknown scenario: {cli.scenario}")
