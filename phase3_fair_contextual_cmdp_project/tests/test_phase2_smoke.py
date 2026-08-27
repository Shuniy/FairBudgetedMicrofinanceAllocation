from __future__ import annotations

from phase2.environment import ContextualMicrofinanceEnv
from phase2.feature_encoder import BorrowerFeatureEncoder
from phase2.models import ContextualEnvironmentConfig
from phase2.policies import HardFairContextualPolicy, LinUCBPolicy
from phase2.runner import run_episode


def test_phase2_episode_runs_without_overspending() -> None:
    encoder = BorrowerFeatureEncoder()
    config = ContextualEnvironmentConfig(
        total_budget=100_000,
        max_rounds=30,
        pool_size=10,
        seed=7,
    )
    env = ContextualMicrofinanceEnv(config, encoder)
    targets = {"women": 0.20, "minority": 0.05}
    policy = HardFairContextualPolicy(
        base_policy=LinUCBPolicy(
            encoder.n_features,
            alpha=0.5,
            budget_aware=True,
            seed=7,
        ),
        target_shares=targets,
        parity_tolerance=0.05,
        seed=7,
    )

    result = run_episode(env, policy, targets, seed=7)

    assert not result.trajectory.empty
    assert result.summary["cumulative_cost"] <= config.total_budget + 1e-8
    assert 0.0 <= result.summary["demographic_parity_gap"] <= 1.0
    assert 0.0 <= result.summary["quota_satisfaction_rate"] <= 1.0


def test_context_dimension_matches_encoder() -> None:
    encoder = BorrowerFeatureEncoder()
    env = ContextualMicrofinanceEnv(
        ContextualEnvironmentConfig(
            total_budget=50_000,
            max_rounds=5,
            pool_size=4,
            seed=1,
        ),
        encoder,
    )
    assert env.context_matrix().shape == (4, encoder.n_features)
