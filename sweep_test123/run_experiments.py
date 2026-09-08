"""Run Modes 1–4 for a named experiment scenario."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import random

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import main as runner

MASTER_SEED = 5726455
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

def generate_scenarios(num_scenarios=10, master_seed=MASTER_SEED):
    """Generate a list of scenarios based on the given options."""
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

# Generate ten scenarios from the parameter choices above.
SCENARIOS = generate_scenarios(num_scenarios=10, master_seed=MASTER_SEED)
print(f"Generated {len(SCENARIOS)} scenarios: {list(SCENARIOS.keys())}", flush=True)

def run(name: str, output_root: Path = ROOT / "outputs") -> None:
    """Run one scenario using main.py's experiment functions."""
    config = SCENARIOS[name]
    destination = Path(output_root) / name
    for mode, label, experiment in (
        ("1", "arm", runner.run_arm),
        ("2", "contextual", runner.run_contextual),
        ("3", "cmdp", runner.run_cmdp),
        ("4", "cmdp_robustness", runner.run_cmdp_sweeps),
    ):
        options = {key: value for key, value in config.items() if key != "arm_budget"}
        if mode == "1":
            options["budget"] = config["arm_budget"]
        if mode != "2":
            options["feedback_delay"] = 0
        argv = ["--mode", mode]
        for key, value in options.items():
            argv.extend(["--" + key.replace("_", "-"), str(value)])
        args = runner.parse_args(argv)
        print(f"Running {name}: {label}", flush=True)
        experiment(args, destination / label)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    for scenario in SCENARIOS:
        run(scenario, args.output_root)


if __name__ == "__main__":
    main()
