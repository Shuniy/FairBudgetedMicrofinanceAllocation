from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from Algorithms.bandit import RandomPolicy
from Algorithms.budgeted_ucb import BudgetedUCBPolicy
from Algorithms.contextual import (
    ContextualEpsilonGreedyPolicy,
    ContextualThompsonSamplingPolicy,
    LinearRidgeModel,
    LinUCBPolicy,
    TrueRewardContextualPolicy,
)
from Algorithms.epsilon_first import EpsilonFirstPolicy
from Algorithms.epsilon_greedy import DecayingEpsilonGreedyPolicy, EpsilonGreedyPolicy
from Algorithms.fairness import HardFairPolicy, SoftFairPolicy
from Algorithms.fractional_kube import FractionalKUBEPolicy
from Algorithms.kube import KUBEPolicy
from Algorithms.reward_per_cost import RewardPerCostPolicy
from Algorithms.thompson_sampling import ThompsonSamplingPolicy
from Algorithms.true_repayment_cost import TrueRewardPerCostPolicy
from Algorithms.ucb import UCBPolicy
from Environments.base_environment import BanditEnvironment
from Environments.contextual_bandit_environment import ContextualBanditEnvironment
from Environments.synthetic_environment_bandits import ArmBanditEnvironment
from evaluate import run_episode
from models.arm_spec import ArmSpec
from models.environment_config import ContextualEnvironmentConfig, EnvironmentConfig
from models.events import FeedbackEvent


def arm_environment(costs, probabilities, budget=20):
    arms = [
        ArmSpec(str(i), str(i), probability, cost)
        for i, (cost, probability) in enumerate(zip(costs, probabilities))
    ]
    return ArmBanditEnvironment(
        arms, EnvironmentConfig(total_budget=budget, max_rounds=10)
    )


def test_ucb_and_budgeted_ucb_scores_use_observed_counts():
    env = arm_environment([2, 4, 8], [0.5, 0.5, 0.5])
    for policy, costs in ((UCBPolicy(3), [1, 1]), (BudgetedUCBPolicy(3), [2, 4])):
        for action, reward, cost in ((0, 1, 2), (0, 0, 2), (1, 1, 4)):
            policy.update(FeedbackEvent(0, reward, cost, action=action))

        assert policy.mean_costs[:2] == pytest.approx([2, 4])
        assert np.isposinf(policy.mean_costs[2])
        scores = policy.scores(env)
        assert scores[:2] == pytest.approx(
            [1.67741002 / costs[0], 2.66510922 / costs[1]]
        )
        assert np.isposinf(scores[2])


def test_budgeted_ucb_and_fractional_kube_are_equivalent():
    env = arm_environment([2, 4, 8], [0.5, 0.5, 0.5], budget=6)
    budgeted = BudgetedUCBPolicy(3, seed=7)
    fractional = FractionalKUBEPolicy(3, seed=7)

    for feedback in (
        None,
        FeedbackEvent(0, 1, 2, action=0),
        FeedbackEvent(1, 0, 4, action=1),
    ):
        if feedback is not None:
            budgeted.update(feedback)
            fractional.update(feedback)
        np.testing.assert_array_equal(budgeted.scores(env), fractional.scores(env))
        for _ in range(5):
            action = budgeted.select_action(env)
            assert action in (0, 1)
            assert action == fractional.select_action(env)

    np.testing.assert_array_equal(budgeted.counts, [1, 1, 0])
    np.testing.assert_array_equal(budgeted.counts, fractional.counts)
    np.testing.assert_array_equal(budgeted.reward_sums, fractional.reward_sums)
    np.testing.assert_array_equal(budgeted.cost_sums, fractional.cost_sums)
    assert budgeted.total_updates == fractional.total_updates == 2


def test_score_ties_distinguish_positive_and_negative_infinity():
    env = arm_environment([2, 2], [1, 1])
    policy = RandomPolicy(2)
    policy.scores = Mock(return_value=np.array([np.inf, -np.inf]))
    assert policy.select_action(env) == 0


@pytest.mark.parametrize(
    "policy_class, settings",
    [
        (EpsilonGreedyPolicy, {"epsilon": 0}),
        (DecayingEpsilonGreedyPolicy, {"initial_epsilon": 0, "minimum_epsilon": 0}),
    ],
)
def test_epsilon_greedy_forces_only_available_unobserved_arms(policy_class, settings):
    policy = policy_class(3, **settings)
    affordable = arm_environment([2, 4, 8], [1, 0, 0])
    policy.update(FeedbackEvent(0, 1, 2, action=0))
    policy.update(FeedbackEvent(1, 0, 4, action=1))
    assert policy.select_action(affordable, [0, 2]) == 2
    assert policy.select_action(affordable, [0, 1]) == 0
    unaffordable = arm_environment([2, 4, 8], [1, 0, 0], budget=6)
    assert policy.scores(unaffordable) == pytest.approx([1, 0, 0])
    assert policy.select_action(unaffordable) == 0


def test_decaying_epsilon_uses_feedback_count_and_floor():
    env = arm_environment([2], [1])
    policy = DecayingEpsilonGreedyPolicy(1, decay_rate=0.5, minimum_epsilon=0.4)
    policy.update(FeedbackEvent(0, 1, 2, action=0))
    policy.scores(env)
    assert policy.epsilon == pytest.approx(2 / 3)
    for round_index in range(1, 4):
        policy.update(FeedbackEvent(round_index, 1, 2, action=0))
    policy.scores(env)
    assert policy.epsilon == 0.4


def test_epsilon_first_spends_through_target_then_exploits():
    env = arm_environment([3, 4], [1, 0])
    policy = EpsilonFirstPolicy(2, exploration_budget_fraction=0.2)
    policy.update(env.step(policy.select_action(env, [0])).observed_feedback[0])

    assert env.spent_budget == 3
    assert policy.scores(env) == pytest.approx([-1, 0])
    policy.update(env.step(policy.select_action(env)).observed_feedback[0])

    assert env.spent_budget == 7
    assert policy.scores(env) == pytest.approx([1 / 3, 0])
    assert policy.select_action(env) == 0


def test_epsilon_first_excludes_unobserved_arms_at_threshold():
    env = arm_environment([4, 3], [1, 0])
    policy = EpsilonFirstPolicy(2, exploration_budget_fraction=0.2, seed=0)
    policy.update(env.step(0).observed_feedback[0])

    assert env.spent_budget == 4
    assert policy.scores(env) == pytest.approx([0.25, -np.inf])
    assert policy.select_action(env) == 0
    assert policy.select_action(env, [1]) == 1


def test_kube_recovers_mixed_counts_and_samples_by_multiplicity():
    env = arm_environment([3.4, 4.3, 11], [0.6, 0.85, 1], budget=10)
    policy = KUBEPolicy(3, exploration_coefficient=0)
    policy.counts[:2] = [10, 20]
    policy.reward_sums[:2] = [6, 17]
    choice = Mock(return_value=1)
    policy.rng = SimpleNamespace(choice=choice)

    # Rounded weights 3 and 4 give 2 * 0.6 + 0.85 = 2.05.
    # Three of arm 0 give 1.8 and two of arm 1 give 1.7.
    assert policy.select_action(env) == 1
    (actions,), kwargs = choice.call_args
    assert dict(zip(actions, kwargs["p"])) == pytest.approx({0: 2 / 3, 1: 1 / 3})
    assert env.feasible_actions() == [0, 1]


def test_kube_initial_exploration_and_rounded_capacity_fallback():
    env = arm_environment([0.6, 0.7], [0, 1], budget=0.9)
    policy = KUBEPolicy(2, exploration_coefficient=0)
    policy.update(FeedbackEvent(0, 0, 0.6, action=0))
    assert policy.select_action(env) == 1
    policy.update(FeedbackEvent(1, 1, 0.7, action=1))

    # No rounded cost fits capacity zero. The fallback uses actual costs.
    assert policy.select_action(env) == 1


def test_thompson_updates_only_observed_arm_beta_parameters():
    policy = ThompsonSamplingPolicy(2)
    for reward in (1, 0, 1):
        policy.update(FeedbackEvent(0, reward, 2, action=1))

    assert policy.alpha == pytest.approx([1, 3])
    assert policy.beta == pytest.approx([1, 2])
    np.testing.assert_array_equal(policy.counts, [0, 3])
    assert policy.mean_rewards == pytest.approx([0, 2 / 3])


def test_reward_per_cost_prefers_unobserved_feasible_arms_then_current_density():
    env = arm_environment([2, 5], [1, 1])
    policy = RewardPerCostPolicy(2)
    policy.update(FeedbackEvent(0, 1, 10, action=0))
    assert policy.select_action(env) == 1

    policy.update(FeedbackEvent(1, 1, 1, action=1))
    assert policy.scores(env) == pytest.approx([0.5, 0.2])
    assert policy.select_action(env) == 0


def test_hard_quota_then_parity_and_soft_quota_bonus():
    env = arm_environment([2, 1], [0.5, 0.5])
    for action in (0, 1, 1, 1):
        env.step(action)
    env.group_opportunity_counts.update({"0": 4, "1": 4})
    base = BudgetedUCBPolicy(2, exploration_coefficient=0)
    hard = HardFairPolicy(base, {"1": 0.7}, parity_tolerance=0.2)
    for action in (0, 1):
        hard.update(FeedbackEvent(0, 1, 2, action=action))

    assert hard.select_action(env) == 1  # Projected quota deficit 0.5.
    env.group_funding_counts.update({"0": 0, "1": 4})
    assert hard.select_action(env) == 0  # Quota met even after funding group 0.
    assert hard.select_action(env, [1]) == 1

    soft = SoftFairPolicy(base, {"0": 0.4}, penalty_lambda=2)
    soft.update(FeedbackEvent(0, 1, 2, action=0))
    assert soft.scores(env) == pytest.approx([1.3, np.inf])
    soft.update(FeedbackEvent(1, 1, 2, action=1))
    assert soft.scores(env) == pytest.approx([1.3, 1])
    assert soft.select_action(env) == 0


def test_ridge_updates_and_linucb_scores():
    model = LinearRidgeModel(2, regularization=2)
    np.testing.assert_array_equal(model.A, [[2, 0], [0, 2]])
    np.testing.assert_array_equal(model.b, [0, 0])
    policy = LinUCBPolicy(2, alpha=0.5, regularization=2, budget_aware=True)
    for context, reward in ((np.array([1, 1]), 1), (np.array([1, 0]), 0)):
        model.update(context, reward)
        policy.update(FeedbackEvent(0, reward, 2, context=context))

    np.testing.assert_array_equal(model.A, [[4, 1], [1, 3]])
    np.testing.assert_array_equal(model.b, [1, 1])
    np.testing.assert_array_equal(policy.model.A, model.A)
    assert model.theta == pytest.approx([2 / 11, 3 / 11])
    env = SimpleNamespace(
        mode="contextual", context_matrix=lambda: np.eye(2),
        current_costs=lambda: np.array([2, 4]),
    )
    assert policy.scores(env) == pytest.approx([
        (2 / 11 + np.sqrt(3 / 11) / 2) / 2,
        (3 / 11 + np.sqrt(4 / 11) / 2) / 4,
    ])


def test_contextual_epsilon_greedy_clips_before_cost_normalization():
    policy = ContextualEpsilonGreedyPolicy(1, epsilon=0, budget_aware=True)
    policy.update(FeedbackEvent(0, 1, 2, context=np.array([1])))
    env = SimpleNamespace(
        mode="contextual", context_matrix=lambda: np.array([[4], [-1]]),
        current_costs=lambda: np.array([2, 4]),
    )
    assert policy.scores(env) == pytest.approx([0.5, 0])


def test_contextual_thompson_uses_one_draw_with_squared_scale():
    policy = ContextualThompsonSamplingPolicy(2, posterior_scale=0.5, budget_aware=True)
    policy.update(FeedbackEvent(0, 1, 2, context=np.array([1, 1])))
    draw = Mock(return_value=np.array([2.0, -1.0]))
    policy.rng = SimpleNamespace(multivariate_normal=draw)
    env = SimpleNamespace(
        mode="contextual", context_matrix=lambda: np.eye(2),
        current_costs=lambda: np.array([2, 4]),
    )

    assert policy.scores(env) == pytest.approx([1, -0.25])
    draw.assert_called_once()
    mean, covariance = draw.call_args.args
    assert mean == pytest.approx([1 / 3, 1 / 3])
    np.testing.assert_allclose(covariance, [[1 / 6, -1 / 12], [-1 / 12, 1 / 6]])


def test_informed_greedy_benchmarks_choose_feasible_true_reward_per_cost():
    arm_env = arm_environment([10, 4, 2], [1, 0.2, 0.12], budget=8)
    contextual_env = SimpleNamespace(
        mode="contextual", current_costs=arm_env.current_costs,
        true_probabilities=arm_env.true_probabilities,
        feasible_actions=arm_env.feasible_actions,
    )
    for policy, env in (
        (TrueRewardPerCostPolicy(3), arm_env),
        (TrueRewardContextualPolicy(2), contextual_env),
    ):
        assert policy.scores(env) == pytest.approx([0.1, 0.05, 0.06])
        assert policy.select_action(env) == 2
    reward_only = TrueRewardContextualPolicy(2, budget_aware=False)
    assert reward_only.select_action(contextual_env) == 1


def test_generic_fairness_wrappers_run_in_arm_mode() -> None:
    arms = [ArmSpec("A", "A", 0.8, 5), ArmSpec("B", "B", 0.6, 5)]
    for policy in (
        HardFairPolicy(RandomPolicy(2, 1), {"A": 0.5, "B": 0.5}, seed=1),
        SoftFairPolicy(RandomPolicy(2, 1), {"A": 0.5}, 1.0, seed=1),
    ):
        result = run_episode(
            ArmBanditEnvironment(
                arms, EnvironmentConfig(total_budget=30, max_rounds=4, seed=1)
            ),
            policy,
            {"A": 0.5, "B": 0.5},
            seed=1,
        )
        assert not result.trajectory.empty
        assert policy.total_updates == result.summary["rounds"]


def test_generic_hard_fairness_runs_in_contextual_mode() -> None:
    env = ContextualBanditEnvironment(
        ContextualEnvironmentConfig(
            total_budget=80_000, max_rounds=4, pool_size=5, seed=2
        )
    )
    policy = HardFairPolicy(
        LinUCBPolicy(env.n_features, alpha=0.5, budget_aware=True, seed=2),
        {"women": 0.20, "minority": 0.05},
        parity_tolerance=0.05,
        seed=2,
    )
    result = run_episode(env, policy, policy.target_shares, seed=2)
    assert result.summary["cumulative_cost"] <= env.config.total_budget + 1e-8
    assert 0.0 <= result.summary["demographic_parity_gap"] <= 1.0


def test_policies_run_directly_against_canonical_environment() -> None:
    arms = [ArmSpec("A", "A", 0.8, 5), ArmSpec("B", "B", 0.6, 5)]
    arm_env = BanditEnvironment.for_arms(
        arms, EnvironmentConfig(total_budget=20, max_rounds=2, seed=4)
    )
    arm_result = run_episode(arm_env, RandomPolicy(arm_env.n_arms, seed=4), seed=4)
    assert arm_result.summary["rounds"] == 2

    contextual_env = BanditEnvironment.for_contextual(
        ContextualEnvironmentConfig(
            total_budget=80_000, max_rounds=2, pool_size=4, seed=4
        )
    )
    contextual_result = run_episode(
        contextual_env,
        LinUCBPolicy(contextual_env.n_features, seed=4),
        seed=4,
    )
    assert contextual_result.summary["rounds"] == 2
