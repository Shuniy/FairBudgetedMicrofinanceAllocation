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
    _baseline_frames_match,
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


def test_sweep_constraint_rates_use_all_episodes() -> None:
    raw = pd.DataFrame({
        "sweep_name": "budget",
        "sweep_value": 1.0,
        "parameter_kind": "capital",
        "policy": "example",
        "is_baseline": True,
        "constraint_satisfied": [1, 0, np.nan],
        "parity_evaluable": [True, True, False],
        "population_parity_evaluable": [True, True, False],
        "population_parity_satisfied": [1, 1, np.nan],
        "population_constraint_satisfied": [1, 0, np.nan],
    })
    original = raw.copy()

    aggregate = aggregate_sweep_results(raw).set_index("metric")

    assert "constraint_satisfied" not in aggregate.index
    for metric, rate in (
        ("constraint_satisfaction_rate", 1 / 3),
        ("population_parity_satisfaction_rate", 2 / 3),
        ("population_constraint_satisfaction_rate", 1 / 3),
    ):
        row = aggregate.loc[metric]
        assert row["metric_mean"] == pytest.approx(rate)
        assert row["n_seeds"] == row["episodes_evaluated"] == 3
        assert row["parity_evaluable_episodes"] == 2
        assert row["population_parity_evaluable_episodes"] == 2
    pd.testing.assert_frame_equal(raw, original)


@pytest.fixture
def baseline_results() -> pd.DataFrame:
    return pd.DataFrame({
        "policy": ["example", "example"],
        "seed": [0, 1],
        "feasible_dp_gap": [0.0, np.nan],
        "population_dp_gap": [0.0, np.nan],
        "feasible_dp_groups": [2, 1],
        "population_dp_groups": [2, 1],
        "parity_evaluable": [True, False],
        "parity_satisfied": [1, np.nan],
        "quota_satisfied": [True, True],
        "constraint_satisfied": [1, np.nan],
        "population_parity_evaluable": [True, False],
        "population_parity_satisfied": [1, np.nan],
        "population_constraint_satisfied": [1, np.nan],
        "funded_none": [1, 1],
        "funded_minority": [1, 0],
        "opportunities_none": [2, 2],
        "opportunities_minority": [2, 0],
        "population_opportunities_none": [3, 3],
        "population_opportunities_minority": [3, 0],
    })


def test_baseline_matches_shuffled_rows_with_undefined_parity(baseline_results) -> None:
    candidate = baseline_results.iloc[::-1]

    assert _baseline_frames_match(baseline_results, candidate) == (True, [])


def test_baseline_reports_different_counts_with_matching_gaps(baseline_results) -> None:
    columns = [
        f"{prefix}_{group}"
        for prefix in ("funded", "opportunities", "population_opportunities")
        for group in ("none", "minority")
    ]
    candidate = baseline_results.copy()
    candidate.loc[0, columns] *= 2

    matches, errors = _baseline_frames_match(baseline_results, candidate)

    assert not matches
    assert set(errors) == set(columns)


def test_baseline_treats_missing_group_counts_as_zero(baseline_results) -> None:
    candidate = baseline_results.assign(population_opportunities_aged=[0, np.nan])

    assert _baseline_frames_match(baseline_results, candidate) == (True, [])

    candidate.loc[1, "population_opportunities_aged"] = 1
    for left, right in ((baseline_results, candidate), (candidate, baseline_results)):
        assert _baseline_frames_match(left, right) == (
            False, ["population_opportunities_aged"]
        )


def test_baseline_reports_different_compliance_outcomes(baseline_results) -> None:
    candidate = baseline_results.copy()
    candidate.loc[0, "population_constraint_satisfied"] = 0

    assert _baseline_frames_match(baseline_results, candidate) == (
        False, ["population_constraint_satisfied"]
    )


def test_baseline_reports_missing_fairness_fields_and_duplicate_rows(baseline_results) -> None:
    historical = baseline_results.drop(columns=["feasible_dp_gap", "parity_satisfied"])
    matches, errors = _baseline_frames_match(historical, historical)
    assert not matches
    for dataset in ("reference", "candidate"):
        assert any(
            dataset in error and "feasible_dp_gap" in error and "parity_satisfied" in error
            for error in errors
        )

    duplicate = pd.concat([baseline_results, baseline_results.iloc[[0]]])
    matches, errors = _baseline_frames_match(duplicate, duplicate)
    assert not matches
    for dataset in ("reference", "candidate"):
        assert any("duplicate policy/seed rows" in error and dataset in error for error in errors)


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
    assert manifest["simulation_version"] == 2
    assert manifest["cases"][0]["policy_configurations"]["portfolio_linucb"][
        "parameters"
    ]["base_shadow_price"] == 0.22
    assert set(artifact.raw_results["feasibility_status"]) <= {
        "opportunity_infeasible",
        "parity_not_evaluable",
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
