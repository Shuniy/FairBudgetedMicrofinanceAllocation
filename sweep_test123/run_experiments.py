from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import types

ROOT = Path(os.getcwd())
OUT = ROOT / "outputs"
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(OUT / ".matplotlib"))
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
sys.path.insert(0, str(ROOT))

from experiment_batch import MASTER_SEED, generate_scenarios, read_plan, sha256, verify_scenario

PLAN_PATH = OUT / "random_experiment_plan.json"
SCENARIOS = read_plan(PLAN_PATH)["scenarios"] if PLAN_PATH.exists() else generate_scenarios(10)

def load_runner():
    source = (ROOT / "main.py").read_text()
    ordinary = "list(range(args.seeds))"
    sweep = "list(range(min(args.seeds, 2) if args.quick_test else args.seeds))"
    assert source.count(ordinary) == 3 and source.count(sweep) == 1
    adapted = source.replace(ordinary, "list(args.seed_values)").replace(
        sweep, "list(args.seed_values[:2] if args.quick_test else args.seed_values)")
    module = types.ModuleType("temporary_experiment_main")
    module.__file__ = str(ROOT / "main.py")
    exec(compile(adapted, module.__file__, "exec"), module.__dict__)
    return module, source, adapted


def run(name, cfg=None, output_root=OUT, plan=None, plan_path=None):
    cfg = SCENARIOS[name] if cfg is None else cfg
    destination = Path(output_root) / name
    destination.mkdir(parents=True, exist_ok=True)
    if plan is not None:
        from experiment_batch import validate_provenance
        validate_provenance(ROOT, plan)
    manifest_path = destination / "experiment_manifest.json"
    if manifest_path.exists():
        old_manifest = json.loads(manifest_path.read_text())
        if old_manifest["parameters"] != cfg or (plan_path and old_manifest.get("plan_sha256") != sha256(plan_path)):
            raise ValueError(f"Refusing mismatched existing scenario: {destination}")
    elif any(destination.iterdir()):
        raise ValueError(f"Refusing unmanifested partial results: {destination}")
    if (destination / "completed.json").exists():
        verify_scenario(destination, cfg, verify_checksums=True)
        print(f"SKIP {name}: completed results verified", flush=True)
        return
    runner, source, adapted = load_runner()
    (destination / "main_runner_snapshot.py").write_text(adapted)
    args = runner.parse_args(["--mode", "5"])
    for key, value in cfg.items():
        if key not in {"seed_start", "arm_budget"}:
            setattr(args, key, value)
    args.seed_values = list(range(cfg["seed_start"], cfg["seed_start"] + cfg["seeds"]))
    args.output = destination
    tracked_sources = [ROOT / "main.py", ROOT / "evaluate.py", ROOT / "robustness.py", ROOT / "metrics.py", ROOT / "simulation.py"]
    for folder in ("Algorithms", "Data", "Environments", "Helpers", "models"):
        tracked_sources.extend((ROOT / folder).glob("*.py"))
    manifest = dict(scenario=name, master_seed=plan["master_seed"] if plan else MASTER_SEED,
        plan_sha256=sha256(plan_path) if plan_path else None,
        environment=plan["environment"] if plan else None,
        runner_sha256=plan["runner_sha256"] if plan else None,
        parameters=cfg, seed_values=args.seed_values,
        parity_tolerance=0.05, contextual_parity_tolerance=0.01, quota_tolerance=0.01,
        full_sweep=True, source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked_sources},
        adapter="Four seed-list expressions changed in temporary main.py copy; original sources unchanged.",
        started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    start = time.monotonic()
    results = {}
    for mode, label, fn in [(1, "arm", runner.run_arm), (2, "contextual", runner.run_contextual), (3, "cmdp", runner.run_cmdp), (4, "cmdp_robustness", runner.run_cmdp_sweeps)]:
        if (destination / f"mode_{mode}_completed.json").exists():
            if mode <= 3:
                import pandas as pd
                results[label] = pd.read_csv(destination / label / "aggregate_results.csv", header=[0, 1], index_col=0)
            print(f"RESUME {name}: mode {mode} already completed", flush=True)
            continue
        phase_args = argparse.Namespace(**vars(args))
        phase_args.mode = str(mode)
        if mode == 1:
            phase_args.budget = cfg["arm_budget"]
        if mode != 2:
            phase_args.feedback_delay = 0
        phase_start = time.monotonic()
        print(f"START {name} mode {mode}: seeds={args.seed_values}, budget={phase_args.budget}", flush=True)
        result = fn(phase_args, destination / label)
        if mode <= 3:
            results[label] = result
        elapsed = time.monotonic() - phase_start
        print(f"DONE {name} mode {mode}: {elapsed:.1f} seconds", flush=True)
        (destination / f"mode_{mode}_completed.json").write_text(json.dumps({"seconds": elapsed}) + "\n")
    runner.combine_aggregate_results(results).to_csv(destination / "combined_aggregate_results.csv", index=False)
    runner.save_combined_report(
        {"contextual": runner.rank_results(results["contextual"], 0.01, 0.01),
         "cmdp": runner.rank_results(results["cmdp"], 0.05, 0.01)},
        destination, arm_aggregate=results["arm"],
        sweep_report=destination / "cmdp_robustness" / "report.html")
    elapsed = time.monotonic() - start
    verification = verify_scenario(destination, cfg)
    from experiment_batch import save_json, artifact_inventory
    save_json(destination / "verification.json", verification)
    save_json(destination / "artifact_checksums.json", artifact_inventory(destination))
    save_json(destination / "completed.json", {"seconds_this_invocation": elapsed,
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "verified": True,
        "phase_episodes": verification["phase_episodes"], "sweep_episodes": verification["sweep_episodes"]})
    print(f"COMPLETE {name}: {elapsed:.1f} seconds", flush=True)


if __name__ == "__main__":
    from experiment_batch import main
    main(ROOT, OUT, run)
