from dataclasses import replace
from math import isnan

import pandas as pd
import pytest

from Algorithms.bandit import RandomPolicy
from Algorithms.cmdp import HardFairCMDPPolicy, OraclePortfolioPolicy
from Algorithms.contextual import TrueRewardContextualPolicy
from Data.borrower import Borrower
from Environments.base_environment import BanditEnvironment
from Environments.cmdp_environment import StatefulMicrofinanceCMDPEnv
from Helpers.enums import Gender
from evaluate import (
    aggregate_results,
    episode_constraint_outcomes,
    rank_results,
    run_cmdp_episode,
    run_episode,
)
from models.arm_spec import ArmSpec
from models.borrower_candidate import BorrowerCandidate
from models.environment_config import (
    CMDPEnvironmentConfig,
    ContextualEnvironmentConfig,
    EnvironmentConfig,
)
from robustness import _parity_opportunity_feasible, _quota_opportunity_feasible
from simulation import SyntheticBorrowerGenerator


def test_arm_episode_reports_allocation_shares_and_zero_funding() -> None:
    arms = [ArmSpec("A", "A", 1.0, 4), ArmSpec("B", "B", 1.0, 8)]
    env = BanditEnvironment.for_arms(arms, EnvironmentConfig(5, 2))
    result = run_episode(env, RandomPolicy(2))
    assert result.summary["allocation_share_B"] == 0
    assert result.summary["allocation_share_gap"] == 1
    assert result.summary["demographic_parity_gap"] == 1
    assert result.trajectory.iloc[0]["allocation_share_gap"] == 1

    env = BanditEnvironment.for_arms(arms, EnvironmentConfig(1, 2))
    summary = run_episode(env, RandomPolicy(2), {"B": 0.50}).summary
    assert summary["allocation_share_gap"] == 0
    assert summary["total_quota_deficit"] == 0.50
    assert not summary["parity_evaluable"]
    assert isnan(summary["constraint_satisfied"])


@pytest.fixture
def fixed_pool(monkeypatch):
    def generate_pool(generator, pool_size):
        applications = (
            dict(age=65, gender=Gender.FEMALE, disability=True,
                 reserved_by_government=True, marginalized_community=True),
            dict(gender=Gender.FEMALE),
            dict(gender=Gender.MALE, marginalized_community=True),
            dict(gender=Gender.MALE),
        )
        pool = []
        for cost, attributes in zip((30, 45, 120, 35), applications):
            application = replace(
                generator._generate_application(),
                **{
                    "age": 35,
                    "disability": False,
                    "reserved_by_government": False,
                    "marginalized_community": False,
                    "requested_loan_amount": cost,
                    **attributes,
                },
            )
            borrower = Borrower(application)
            pool.append(BorrowerCandidate(borrower, generator.encoder.transform(borrower), 1.0))
        assert len(pool) == pool_size
        return pool

    monkeypatch.setattr(SyntheticBorrowerGenerator, "generate_pool", generate_pool)


def test_contextual_counts_the_terminal_pool_before_affordability(fixed_pool) -> None:
    env = BanditEnvironment.for_contextual(
        ContextualEnvironmentConfig(total_budget=40, max_rounds=3, pool_size=4)
    )
    report = run_episode(
        env, TrueRewardContextualPolicy(env.n_features), parity_tolerance=0.75
    )
    first = report.trajectory.iloc[0]
    final = report.summary
    assert len(report.trajectory) == 1
    assert first["population_opportunities_disabled"] == 1
    assert final["population_opportunities_disabled"] == 2
    assert final["opportunities_disabled"] == 1
    assert final["funded_disabled"] == 1
    assert final["population_dp_gap"] == 0.50
    assert final["feasible_dp_gap"] == 1.0
    assert final["feasible_dp_groups"] == 2
    assert final["population_dp_groups"] == 4
    assert final["population_constraint_satisfied"] == 1
    assert final["constraint_satisfied"] == 0
    assert final["opportunities_minority"] == 0
    assert isnan(final["funding_rate_minority"])


def test_cmdp_batch_counts_each_applicant_in_one_primary_group(fixed_pool) -> None:
    env = StatefulMicrofinanceCMDPEnv(CMDPEnvironmentConfig(
        initial_capital=100, minimum_liquidity_reserve=10, minimum_cost=1,
        max_rounds=1, pool_size=4, max_loans_per_round=2,
    ))
    assert len(env.current_pool[0].borrower.protected_groups) == 5
    assert env.current_pool[0].fairness_group == "disabled"
    base = OraclePortfolioPolicy(env.encoder.n_features, state_aware=False)
    policy = HardFairCMDPPolicy(base, {"minority": 0.5}, parity_tolerance=1.0)
    assert policy.select_action(env, [2]) is None
    report = run_cmdp_episode(env, policy, {"minority": 0.5, "disabled": 0.5}, 1.0, 0)
    assert report.funding_log["fairness_group"].tolist() == ["disabled", "none"]
    expected = {
        "disabled": (1, 1, 1),
        "women": (0, 1, 1),
        "minority": (0, 0, 1),
        "none": (1, 1, 1),
    }
    for row in (report.trajectory.iloc[0], report.summary):
        for group, counts in expected.items():
            assert (
                row[f"funded_{group}"],
                row[f"opportunities_{group}"],
                row[f"population_opportunities_{group}"],
            ) == counts
            funded, feasible, population = counts
            assert 0 <= funded <= feasible <= population
        assert row["allocation_share_disabled"] == 0.50
        assert row["total_quota_deficit"] == 0.50
        assert row["quota_satisfaction_rate"] == 0.50
        assert row["population_dp_gap"] == row["feasible_dp_gap"] == 1.0


def test_episode_compliance_differs_from_mean_thresholds() -> None:
    summaries = pd.DataFrame([
        {
            "policy": "example",
            "cumulative_reward": 1,
            "reward_per_cost": 1,
            "demographic_parity_gap": gap,
            "total_quota_deficit": deficit,
            **episode_constraint_outcomes(gap, deficit, 0.05, 0.01),
        }
        for gap, deficit in zip([0.01, 0.03, 0.08], [0.0, 0.02, 0.0])
    ])
    ranked = rank_results(aggregate_results(summaries), 0.05).iloc[0]
    assert ranked["demographic_parity_gap_mean"] == pytest.approx(0.04)
    assert ranked["total_quota_deficit_mean"] == pytest.approx(0.02 / 3)
    assert ranked["mean_thresholds_met"]
    assert not ranked["fairness_feasible"]
    assert ranked["parity_satisfaction_rate"] == pytest.approx(2 / 3)
    assert ranked["quota_constraint_satisfaction_rate"] == pytest.approx(2 / 3)
    assert ranked["constraint_satisfaction_rate"] == pytest.approx(1 / 3)

    undefined = episode_constraint_outcomes(float("nan"), 0.0, 0.05, 0.01)
    assert isnan(undefined["parity_satisfied"])
    assert isnan(undefined["constraint_satisfied"])
    summaries.loc[3] = {
        **summaries.iloc[0].to_dict(), **undefined, "demographic_parity_gap": float("nan")
    }
    aggregate = aggregate_results(summaries).loc["example"]
    assert aggregate["episodes_evaluated", ""] == 4
    assert aggregate["parity_evaluable_episodes", ""] == 3
    assert aggregate["constraint_satisfaction_rate", ""] == 0.25


def test_opportunity_screens_check_volume_rounding_and_undefined_parity() -> None:
    opportunities = {"A": 5, "B": 5}
    assert not _quota_opportunity_feasible(opportunities, 3, {"A": 0.5, "B": 0.5})
    assert _quota_opportunity_feasible(opportunities, 3, {"A": 0.5, "B": 0.5}, 0.17)
    assert not _quota_opportunity_feasible(opportunities, 0, {"A": 0.5}, 0.01)
    assert _quota_opportunity_feasible(opportunities, 0, {}, 0.01)
    assert not _quota_opportunity_feasible(opportunities, 11, {})
    assert not _parity_opportunity_feasible(opportunities, 3, 0.10)
    assert _parity_opportunity_feasible(opportunities, 3, 0.20)
    assert _parity_opportunity_feasible({"A": 5, "B": 0}, 2, 0.20) is None
    # Both possible allocations have a gap of 2/3, despite overlapping totals.
    assert not _parity_opportunity_feasible({"A": 1, "B": 3}, 2, 0.50)

    # At one loan, quota requires B while the parity threshold allows only A.
    assert _quota_opportunity_feasible({"A": 10, "B": 1}, 1, {"B": 1.0})
    assert _parity_opportunity_feasible({"A": 10, "B": 1}, 1, 0.10)
