from __future__ import annotations

import json

import numpy as np
import pandas as pd

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
    manifest = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert manifest["policies"] == ["portfolio_linucb"]
    assert manifest["seeds"] == [0, 1]
    assert manifest["horizon"] == 3
    assert set(artifact.raw_results["feasibility_status"]) <= {
        "opportunity_infeasible",
        "constraint_satisfied",
        "policy_constraint_violation",
    }
