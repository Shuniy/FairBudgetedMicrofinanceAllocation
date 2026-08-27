from __future__ import annotations

import numpy as np

from Algorithms.cmdp import (
    LagrangianFairCMDPPolicy,
    PortfolioAwareLinUCBPolicy,
)
from Data.feature_encoder import BorrowerFeatureEncoder
from Environments.cmdp_environment import StatefulMicrofinanceCMDPEnv
from evaluate import compare_cmdp_policies, run_cmdp_episode, save_cmdp_results
from models.environment_config import CMDPEnvironmentConfig
from models.fairness_config import LagrangianConfig
from simulation import SyntheticBorrowerGenerator


def cmdp_config(**overrides) -> CMDPEnvironmentConfig:
    values = {
        "initial_capital": 200_000.0,
        "max_rounds": 12,
        "pool_size": 8,
        "max_loans_per_round": 1,
        "repayment_delay_rounds": 2,
        "seed": 3,
    }
    values.update(overrides)
    return CMDPEnvironmentConfig(**values)


def test_cmdp_reuses_parent_generator_and_recycles_at_maturity() -> None:
    encoder = BorrowerFeatureEncoder(include_protected_attributes=False)
    config = cmdp_config()
    env = StatefulMicrofinanceCMDPEnv(config, encoder)
    assert isinstance(env.generator, SyntheticBorrowerGenerator)

    action = env.feasible_actions()[0]
    result = env.fund(action, env.current_pool[action].context.copy())
    after_funding = env.available_capital
    env.advance_round()
    assert np.isclose(env.available_capital, after_funding)
    advance = env.advance_round()
    assert len(advance.matured_feedback) == 1
    feedback = advance.matured_feedback[0]
    expected = after_funding + (
        result.principal * config.recycle_fraction
        if feedback.reward == 1.0
        else result.principal * config.default_recovery_fraction
    )
    assert np.isclose(env.available_capital, expected)


def test_cmdp_episode_never_overspends_liquid_capital() -> None:
    encoder = BorrowerFeatureEncoder(include_protected_attributes=False)
    config = cmdp_config(
        initial_capital=150_000,
        max_rounds=18,
        max_loans_per_round=2,
        repayment_delay_rounds=4,
        seed=4,
    )
    policy = PortfolioAwareLinUCBPolicy(encoder.n_features, seed=4)
    episode = run_cmdp_episode(
        StatefulMicrofinanceCMDPEnv(config, encoder),
        policy,
        target_shares={"women": 0.20, "minority": 0.10},
        parity_tolerance=0.10,
        seed=4,
    )
    assert not episode.trajectory.empty
    assert (episode.trajectory["available_capital"] >= -1e-9).all()
    assert episode.summary["capital_turnover"] >= 0.0


def test_lagrange_multipliers_are_nonnegative_and_bounded() -> None:
    encoder = BorrowerFeatureEncoder(include_protected_attributes=False)
    config = cmdp_config(max_rounds=15, repayment_delay_rounds=3, seed=5)
    lagrangian = LagrangianConfig(
        target_shares={"women": 0.25, "minority": 0.12},
        parity_tolerance=0.05,
        max_lambda=2.0,
        warmup_rounds=2,
    )
    policy = LagrangianFairCMDPPolicy(
        PortfolioAwareLinUCBPolicy(encoder.n_features, seed=5),
        lagrangian,
        seed=5,
    )
    run_cmdp_episode(
        StatefulMicrofinanceCMDPEnv(config, encoder),
        policy,
        target_shares=lagrangian.target_shares,
        parity_tolerance=lagrangian.parity_tolerance,
        seed=5,
    )
    assert 0.0 <= policy.lambda_quota <= lagrangian.max_lambda
    assert 0.0 <= policy.lambda_parity <= lagrangian.max_lambda
    assert policy.lambda_history


def test_cmdp_results_include_portfolio_audit_files(tmp_path) -> None:
    encoder = BorrowerFeatureEncoder(include_protected_attributes=False)
    config = cmdp_config(max_rounds=4, pool_size=4)
    policies = {
        "portfolio": lambda dimensions, seed: PortfolioAwareLinUCBPolicy(
            dimensions, seed=seed
        )
    }
    summaries, trajectories, funding, maturities = compare_cmdp_policies(
        config,
        policies,
        {"women": 0.20},
        0.10,
        [0],
        encoder,
    )
    aggregate = save_cmdp_results(
        summaries,
        trajectories,
        funding,
        maturities,
        tmp_path,
    )
    assert ("capital_turnover", "mean") in aggregate.columns
    for filename in (
        "episode_summaries.csv",
        "trajectories.csv",
        "aggregate_results.csv",
        "funding_log.csv",
        "maturity_log.csv",
    ):
        assert (tmp_path / filename).exists()
