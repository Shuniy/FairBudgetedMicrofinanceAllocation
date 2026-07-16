from __future__ import annotations

from Algorithms.bandit import RandomPolicy
from Algorithms.contextual import LinUCBPolicy
from Algorithms.fairness import HardFairPolicy, SoftFairPolicy
from Environments.base_environment import BanditEnvironment
from Environments.contextual_bandit_environment import ContextualBanditEnvironment
from Environments.synthetic_environment_bandits import ArmBanditEnvironment
from evaluate import run_episode
from models.arm_spec import ArmSpec
from models.environment_config import ContextualEnvironmentConfig, EnvironmentConfig


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
