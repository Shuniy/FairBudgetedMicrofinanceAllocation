from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from Data.feature_encoder import BorrowerFeatureEncoder
from evaluate import compare_cmdp_policies
from main import DEFAULT_TARGET_SHARES, build_cmdp_policy_factories
from models.environment_config import CMDPEnvironmentConfig
from robustness import (
    CMDPSweepCase,
    aggregate_sweep_results,
    run_cmdp_sweep,
)


def test_lagrangian_strength_is_explicit_and_defaults_to_phase3_value() -> None:
    default = build_cmdp_policy_factories(DEFAULT_TARGET_SHARES, 0.05)[
        "cmdp_fair_lagrangian_linucb"
    ](8, 0)
    ablated = build_cmdp_policy_factories(
        DEFAULT_TARGET_SHARES,
        0.05,
        lagrangian_strength_scale=0.0,
    )["cmdp_fair_lagrangian_linucb"](8, 0)
    assert default.config.strength_scale == 1.0
    assert ablated.config.strength_scale == 0.0


def test_tidy_sweep_aggregation_includes_std_and_ci95() -> None:
    raw = pd.DataFrame(
        {
            "sweep_name": ["budget", "budget"],
            "sweep_value": [1.0, 1.0],
            "parameter_kind": ["capital", "capital"],
            "policy": ["phase3", "phase3"],
            "is_baseline": [True, True],
            "eventual_cumulative_reward": [10.0, 14.0],
        }
    )
    aggregate = aggregate_sweep_results(raw)
    reward = aggregate.loc[
        aggregate["metric"] == "eventual_cumulative_reward"
    ].iloc[0]
    assert reward["metric_mean"] == 12.0
    assert reward["metric_std"] == np.sqrt(8.0)
    assert reward["metric_ci95_low"] < 12.0 < reward["metric_ci95_high"]
    assert reward["n_seeds"] == 2


def test_mode5_default_case_matches_standard_cmdp_and_saves_outputs(tmp_path) -> None:
    # A rerun must retire the duplicate left by earlier versions.
    (tmp_path / "episode_summaries.csv").write_text("legacy duplicate\n")
    encoder = BorrowerFeatureEncoder(include_protected_attributes=False)
    config = CMDPEnvironmentConfig(
        initial_capital=120_000,
        max_rounds=3,
        pool_size=5,
        max_loans_per_round=1,
        repayment_delay_rounds=2,
    )
    all_factories = build_cmdp_policy_factories(DEFAULT_TARGET_SHARES, 0.05)
    factories = {"portfolio_linucb": all_factories["portfolio_linucb"]}
    standard, _, _, _ = compare_cmdp_policies(
        config,
        factories,
        DEFAULT_TARGET_SHARES,
        0.05,
        [0, 1],
        encoder,
    )
    artifact = run_cmdp_sweep(
        "budget",
        [
            CMDPSweepCase(
                value=1.0,
                env_config=config,
                policy_factories=factories,
                target_shares=DEFAULT_TARGET_SHARES,
                parity_tolerance=0.05,
                parameter_kind="initial_capital_multiplier",
                parameter_details={"budget_multiplier": 1.0},
                is_baseline=True,
            )
        ],
        [0, 1],
        tmp_path,
        config,
        quota_tolerance=0.01,
        baseline_value=1.0,
        encoder=encoder,
    )
    columns = [
        "policy",
        "seed",
        "cumulative_reward",
        "cumulative_disbursed",
        "demographic_parity_gap",
        "quota_satisfaction_rate",
    ]
    pd.testing.assert_frame_equal(
        standard[columns].sort_values(["policy", "seed"]).reset_index(drop=True),
        artifact.raw_results[columns]
        .sort_values(["policy", "seed"])
        .reset_index(drop=True),
    )
    for filename in (
        "raw_results.csv",
        "aggregate_results.csv",
        "config.json",
        "summary.json",
        "trajectories.csv",
        "funding_log.csv",
        "maturity_log.csv",
    ):
        assert (tmp_path / filename).exists()
    assert not (tmp_path / "episode_summaries.csv").exists()
    pd.testing.assert_frame_equal(
        pd.read_csv(tmp_path / "raw_results.csv"),
        artifact.raw_results,
        check_dtype=False,
    )
    manifest = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert manifest["policies"] == ["portfolio_linucb"]
    assert manifest["seeds"] == [0, 1]
    assert manifest["horizon"] == 3
    assert set(artifact.raw_results["feasibility_status"]) <= {
        "opportunity_infeasible",
        "constraint_satisfied",
        "policy_constraint_violation",
    }


@pytest.mark.parametrize("custom_output", [False, True])
def test_all_reuses_phase_paths_and_links_reports(tmp_path, monkeypatch, custom_output):
    import main as runner

    monkeypatch.chdir(tmp_path)
    cli = ["--mode", "all", "--quick-test", "--rounds", "2", "--seeds", "1",
           "--pool-size", "4", "--batch-size", "1"]
    output = Path("custom") if custom_output else Path("outputs")
    if custom_output:
        cli.extend(["--output", str(output)])
    args = runner.parse_args(cli)
    scenario_outputs = []
    monkeypatch.setattr(runner, "run_all_scenarios", scenario_outputs.append)
    # Start with actual standalone results and verify the suite replaces them
    # in place, rather than leaving a stale table beside a second result tree.
    runner.run_arm(args, output / "arm")
    summary = output / "arm" / "episode_summaries.csv"
    summary.write_text("stale result\n")
    runner.run_all(args)

    assert scenario_outputs == [output / "all_scenarios"]
    assert "policy" in pd.read_csv(summary).columns
    assert not (output / "all").exists()
    for phase in ("arm", "contextual", "cmdp"):
        assert (output / phase / "episode_summaries.csv").exists()
    for sweep in ("budget", "group_imbalance", "fairness_strength"):
        directory = output / "cmdp_robustness" / "sweeps" / sweep
        assert (directory / "raw_results.csv").exists()
        assert not (directory / "episode_summaries.csv").exists()
    report = (output / "report.html").read_text()
    for relative in ("arm/aggregate_results.csv", "contextual/report.html",
                     "cmdp/report.html", "cmdp_robustness/report.html",
                     "combined_aggregate_results.csv", "run_manifest.json"):
        assert relative in report
        assert (output / relative).exists()
