from __future__ import annotations

import random
import subprocess
import sys
from pathlib import Path

import pandas as pd

from sweep_test123 import run_experiments


def test_scenarios_are_repeatable_and_have_distinct_seed_ranges():
    state = random.getstate()
    scenarios = run_experiments.generate_scenarios()
    assert scenarios == run_experiments.SCENARIOS
    assert random.getstate() == state
    seed_sets = [set(range(c["seed_start"], c["seed_start"] + c["seeds"]))
                 for c in scenarios.values()]
    assert len(set.union(*seed_sets)) == sum(map(len, seed_sets))


def test_scenario_cli_runs_all_phases_with_configured_seeds(tmp_path, monkeypatch):
    config = dict(run_experiments.SCENARIOS["scenario_1"])
    config.update(seeds=2, rounds=3, pool_size=4, batch_size=1)
    monkeypatch.setitem(run_experiments.SCENARIOS, "scenario_1", config)
    monkeypatch.setattr(sys, "argv", ["run_experiments.py", "scenario_1",
                                    "--output-root", str(tmp_path)])
    run_experiments.main()

    destination = tmp_path / "scenario_1"
    expected_seeds = {config["seed_start"], config["seed_start"] + 1}
    for phase, policy_count in (("arm", 13), ("contextual", 14), ("cmdp", 12)):
        raw = pd.read_csv(destination / phase / "episode_summaries.csv")
        assert set(raw.seed) == expected_seeds
        assert len(raw) == policy_count * config["seeds"]
        assert not raw.duplicated(["policy", "seed"]).any()
        assert (destination / phase / "aggregate_results.csv").is_file()
    for sweep, rows_per_seed in (("budget", 60), ("group_imbalance", 60),
                                 ("fairness_strength", 36)):
        folder = destination / "cmdp_robustness" / "sweeps" / sweep
        raw = pd.read_csv(folder / "raw_results.csv")
        assert set(raw.seed) == expected_seeds
        assert len(raw) == rows_per_seed * config["seeds"]
        assert (folder / "aggregate_results.csv").is_file()
    assert {p.name for p in tmp_path.iterdir()} == {"scenario_1"}
    assert {p.name for p in destination.iterdir()} == {
        "arm", "contextual", "cmdp", "cmdp_robustness"
    }


def test_cli_works_outside_repository_and_rejects_unknown_scenario(tmp_path):
    script = Path(run_experiments.__file__).resolve()
    help_result = subprocess.run([sys.executable, str(script), "--help"],
                                 cwd=tmp_path, capture_output=True, text=True)
    assert help_result.returncode == 0, help_result.stderr
    assert "scenario_10" in help_result.stdout
    invalid = subprocess.run([sys.executable, str(script), "unknown"],
                             cwd=tmp_path, capture_output=True, text=True)
    assert invalid.returncode == 2
    assert "invalid choice" in invalid.stderr
    assert not list(tmp_path.iterdir())
