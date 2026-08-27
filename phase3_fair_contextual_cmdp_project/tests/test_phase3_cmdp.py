from __future__ import annotations

import numpy as np

from phase2.feature_encoder import BorrowerFeatureEncoder
from phase3.environment import StatefulMicrofinanceCMDPEnv
from phase3.models import CMDPEnvironmentConfig, LagrangianConfig
from phase3.policies import (
    LagrangianFairCMDPPolicy,
    PortfolioAwareLinUCBPolicy,
)
from phase3.runner import run_episode


def test_capital_recycles_only_when_repayment_matures() -> None:
    encoder = BorrowerFeatureEncoder(include_protected_attributes=False)
    config = CMDPEnvironmentConfig(
        initial_capital=200_000,
        max_rounds=20,
        pool_size=8,
        max_loans_per_round=1,
        repayment_delay_rounds=2,
        seed=3,
    )
    env = StatefulMicrofinanceCMDPEnv(config, encoder)
    action = env.feasible_actions()[0]
    context = env.current_pool[action].context.copy()
    result = env.fund(action, context)
    after_funding = env.available_capital
    assert after_funding < config.initial_capital
    env.advance_round()  # round 1: not due yet
    assert np.isclose(env.available_capital, after_funding)
    advance = env.advance_round()  # round 2: loan matures
    assert len(advance.matured_feedback) == 1
    feedback = advance.matured_feedback[0]
    expected = after_funding + (
        result.principal * config.recycle_fraction
        if feedback.reward == 1.0
        else result.principal * config.default_recovery_fraction
    )
    assert np.isclose(env.available_capital, expected)


def test_cmdp_never_lends_more_liquid_capital_than_available() -> None:
    encoder = BorrowerFeatureEncoder(include_protected_attributes=False)
    config = CMDPEnvironmentConfig(
        initial_capital=250_000,
        max_rounds=35,
        pool_size=12,
        max_loans_per_round=3,
        repayment_delay_rounds=5,
        seed=4,
    )
    policy = PortfolioAwareLinUCBPolicy(encoder.n_features, seed=4)
    episode = run_episode(
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
    config = CMDPEnvironmentConfig(
        initial_capital=250_000,
        max_rounds=30,
        pool_size=12,
        max_loans_per_round=2,
        repayment_delay_rounds=4,
        seed=5,
    )
    lag_config = LagrangianConfig(
        target_shares={"women": 0.25, "minority": 0.12},
        parity_tolerance=0.05,
        max_lambda=2.0,
        warmup_rounds=2,
    )
    policy = LagrangianFairCMDPPolicy(
        PortfolioAwareLinUCBPolicy(encoder.n_features, seed=5),
        lag_config,
        seed=5,
    )
    run_episode(
        StatefulMicrofinanceCMDPEnv(config, encoder),
        policy,
        target_shares=lag_config.target_shares,
        parity_tolerance=lag_config.parity_tolerance,
        seed=5,
    )
    assert 0.0 <= policy.lambda_quota <= lag_config.max_lambda
    assert 0.0 <= policy.lambda_parity <= lag_config.max_lambda
    assert policy.lambda_history
