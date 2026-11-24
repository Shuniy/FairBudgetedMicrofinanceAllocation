from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from Algorithms.cmdp import (
    ContextOnlyBudgetedLinUCBPolicy,
    HardFairCMDPPolicy,
    LagrangianFairCMDPPolicy,
    OraclePortfolioPolicy,
    PortfolioAwareEpsilonGreedyPolicy,
    PortfolioAwareLinUCBPolicy,
    PortfolioAwareThompsonPolicy,
    RandomCMDPPolicy,
)
from Data.feature_encoder import BorrowerFeatureEncoder
from Environments.cmdp_environment import StatefulMicrofinanceCMDPEnv
from evaluate import compare_cmdp_policies, run_cmdp_episode, save_cmdp_results
from models.environment_config import CMDPEnvironmentConfig
from models.events import PendingLoan
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


@pytest.mark.parametrize("probability, returned_fraction", [(1.0, 0.75), (0.0, 0.25)])
def test_cmdp_reuses_parent_generator_and_recycles_at_maturity(
    probability, returned_fraction
) -> None:
    encoder = BorrowerFeatureEncoder(include_protected_attributes=False)
    config = cmdp_config(recycle_fraction=0.75, default_recovery_fraction=0.25)
    env = StatefulMicrofinanceCMDPEnv(config, encoder)
    assert isinstance(env.generator, SyntheticBorrowerGenerator)

    action = env.feasible_actions()[0]
    env.current_pool[action] = replace(
        env.current_pool[action], true_repayment_probability=probability
    )
    context = env.current_pool[action].context.copy()
    result = env.fund(action, context)
    after_funding = env.available_capital
    env.advance_round()
    assert np.isclose(env.available_capital, after_funding)
    advance = env.advance_round()
    assert len(advance.matured_feedback) == 1
    feedback = advance.matured_feedback[0]
    np.testing.assert_array_equal(feedback.learning_context, context)
    assert feedback.reward == probability
    expected_return = result.principal * returned_fraction
    assert env.available_capital == pytest.approx(after_funding + expected_return)
    assert env.cumulative_default_loss == pytest.approx(
        (result.principal - expected_return) * (1 - probability)
    )


def test_cmdp_terminal_advance_observes_maturities_and_preserves_pending_reward(monkeypatch):
    monkeypatch.setattr(
        SyntheticBorrowerGenerator, "true_repayment_probability",
        staticmethod(lambda borrower: 1.0),
    )
    env = StatefulMicrofinanceCMDPEnv(
        cmdp_config(max_rounds=2, repayment_delay_rounds=2, recycle_fraction=0.5)
    )
    policy = RandomCMDPPolicy(env.encoder.n_features)
    episode = run_cmdp_episode(env, policy, {}, 1.0, seed=0)

    assert episode.summary["loans_funded"] == 2
    assert episode.summary["cumulative_reward"] == 2
    assert episode.summary["observed_reward_by_horizon"] == 1
    assert episode.summary["pending_loans"] == 1
    assert policy.total_updates == 0
    assert episode.summary["final_available_capital"] == pytest.approx(
        env.config.initial_capital - sum(episode.funding_log["principal"])
        + episode.funding_log.iloc[0]["principal"] / 2
    )


def portfolio_state():
    env = StatefulMicrofinanceCMDPEnv(cmdp_config(initial_capital=100, minimum_cost=1))
    env.available_capital = 50
    env.cumulative_default_loss = 25
    env.pending_loans = [
        PendingLoan("past", "A", 25, 0, 2, 1, 1, np.array([1])),
        PendingLoan("later", "B", 10, 0, 3, 1, 1, np.array([1])),
    ]
    env.current_pool = [
        SimpleNamespace(
            context=np.array([x]), cost=cost, fairness_group=group,
            true_repayment_probability=probability,
        )
        for x, cost, group, probability in (
            (1, 5, "A", 0.55), (2, 10, "B", 0.9), (3, 60, "A", 0.95)
        )
    ]
    return env


def test_portfolio_linucb_epsilon_greedy_and_context_only_scores():
    env = portfolio_state()
    linucb = PortfolioAwareLinUCBPolicy(1, alpha=0.5, regularization=3)
    greedy = PortfolioAwareEpsilonGreedyPolicy(1, epsilon=0, regularization=3)
    context_only = ContextOnlyBudgetedLinUCBPolicy(1, alpha=0.5, regularization=3)
    for policy in (linucb, greedy, context_only):
        policy.update(np.array([1]), reward=1, cost=5)

    # 0.22 * (1 + 1.4 * 0.50 + 0.60 * 0.25) / (1 + 0.80 * 0.25).
    shadow_price = 0.33916666666666667
    assert linucb.capital_shadow_price(env) == pytest.approx(shadow_price)
    assert linucb.scores(env) == pytest.approx([
        0.5 - shadow_price / 2, 1 - shadow_price, 1.5 - 6 * shadow_price,
    ])
    assert greedy.scores(env) == pytest.approx([
        0.25 - shadow_price / 2, 0.5 - shadow_price, 0.75 - 6 * shadow_price,
    ])
    assert context_only.scores(env) == pytest.approx([1, 1, 0.25])

    env.available_capital = 150
    env.cumulative_default_loss = 200
    assert linucb.capital_shadow_price(env) == pytest.approx(0.29333333333333333)


def test_portfolio_thompson_scores_and_informed_heuristic_configuration():
    env = portfolio_state()
    thompson = PortfolioAwareThompsonPolicy(1, posterior_scale=0.3, regularization=3)
    thompson.update(np.array([1]), reward=1, cost=5)
    draw = Mock(return_value=np.array([0.3]))
    thompson.rng = SimpleNamespace(multivariate_normal=draw)
    assert thompson.scores(env) == pytest.approx([0.113, 0.226, -1.344])
    draw.assert_called_once()
    mean, covariance = draw.call_args.args
    assert mean == pytest.approx([0.25])
    np.testing.assert_allclose(covariance, [[0.0225]])

    informed = OraclePortfolioPolicy(1)
    assert informed.scores(env) == pytest.approx([0.363, 0.526, -1.294])
    assert informed.select_action(env) == 1
    density = OraclePortfolioPolicy(1, state_aware=False)
    assert density.scores(env) == pytest.approx([1.10, 0.90, 0.95 / 6])
    assert density.select_action(env) == 0


def test_hard_cmdp_fairness_restricts_to_affordable_candidates():
    env = portfolio_state()
    policy = HardFairCMDPPolicy(OraclePortfolioPolicy(1), {"A": 0.5})
    assert env.feasible_actions() == [0, 1]
    assert policy.select_action(env, [0, 1, 2]) == 0
    assert policy.select_action(env, [1, 2]) == 1
    assert policy.select_action(env, [2]) is None


def test_cmdp_score_ties_distinguish_positive_and_negative_infinity():
    env = portfolio_state()
    policy = RandomCMDPPolicy(1)
    policy.scores = Mock(return_value=np.array([np.inf, -np.inf, 1]))
    assert policy.select_action(env) == 0


def test_lagrangian_scores_normalize_and_use_projected_shortfalls():
    env = portfolio_state()
    env.group_funding_counts.update(A=3, B=1)
    env.group_opportunity_counts.update(A=10, B=10)
    base = PortfolioAwareLinUCBPolicy(1)
    base.scores = Mock(return_value=np.array([3, 1, 100]))
    config = LagrangianConfig(
        target_shares={"A": 0.5, "B": 0.5}, parity_tolerance=0.1,
        initial_lambda_quota=0.4, initial_lambda_parity=0.5,
        strength_scale=2, warmup_rounds=1,
    )
    policy = LagrangianFairCMDPPolicy(base, config)
    assert policy.scores(env) == pytest.approx([1, 0, 0])

    env.round_index = 1
    # Funding A yields quota deficit 0.3 and parity excess 0.2.
    # Funding B yields quota deficit 0.1 and no parity excess.
    assert policy.scores(env) == pytest.approx([0.56, -0.08, 0])
    no_penalty = LagrangianFairCMDPPolicy(base, replace(config, strength_scale=0))
    assert no_penalty.scores(env) == pytest.approx([1, 0, 0])
    no_penalty.on_round_end(env)
    assert no_penalty.lambda_quota > config.initial_lambda_quota
    assert no_penalty.lambda_parity > config.initial_lambda_parity

    base.scores.return_value = np.array([4, 4, 100])
    assert no_penalty.scores(env) == pytest.approx([0.5, 0.5, 0])


def test_lagrangian_updates_use_signed_excess_and_project_both_bounds():
    env = portfolio_state()
    env.group_funding_counts.update(A=3, B=1)
    env.group_opportunity_counts.update(A=10, B=10)
    config = LagrangianConfig(
        target_shares={"A": 0.5, "B": 0.5}, quota_tolerance=0.10,
        parity_tolerance=0.15, dual_learning_rate=1, max_lambda=0.30,
        warmup_rounds=1,
    )
    policy = LagrangianFairCMDPPolicy(PortfolioAwareLinUCBPolicy(1), config)
    policy.on_round_end(env)
    assert policy.lambda_history[-1] == pytest.approx((0.20, 0.20))

    env.round_index = 1
    policy.on_round_end(env)
    assert policy.lambda_history[-1] == pytest.approx((0.30, 0.25))
    env.round_index = 2
    for _ in range(2):
        policy.on_round_end(env)
        env.round_index += 1
    assert policy.lambda_history[-1] == pytest.approx((0.30, 0.30))

    env.group_funding_counts.update(A=2, B=2)
    policy.on_round_end(env)
    assert policy.lambda_history[-1] == pytest.approx((0.20, 0.15))
    for _ in range(3):
        env.round_index += 1
        policy.on_round_end(env)
    assert policy.lambda_history[-1] == pytest.approx((0.0, 0.0))


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
