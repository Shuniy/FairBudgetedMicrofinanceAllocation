from __future__ import annotations

import numpy as np
import pytest

from Data.feature_encoder import BorrowerFeatureEncoder
from Environments.base_environment import BanditEnvironment, BaseBudgetEnvironment
from Environments.contextual_bandit_environment import (
    ContextualBanditEnvironment,
    ContextualMicrofinanceEnv,
)
from Environments.synthetic_environment_bandits import (
    ArmBanditEnvironment,
    SyntheticEnvironment,
)
from models.arm_spec import ArmSpec
from models.environment_config import ContextualEnvironmentConfig, EnvironmentConfig


def test_arm_environment_is_seeded_and_never_overspends() -> None:
    arms = [
        ArmSpec("A", "A", 0.8, 12.0, 1.2),
        ArmSpec("B", "B", 0.6, 8.0, 0.8),
    ]
    config = EnvironmentConfig(total_budget=40, max_rounds=10, seed=7)
    first = ArmBanditEnvironment(arms, config)
    second = ArmBanditEnvironment(arms, config)
    assert np.allclose(first.current_costs(), second.current_costs())
    while not first.done:
        first.step(first.feasible_actions()[0])
    assert first.spent_budget <= config.total_budget + 1e-8
    assert sum(first.group_opportunity_counts.values()) >= first.total_allocations


def test_arm_opportunities_only_include_affordable_groups() -> None:
    env = ArmBanditEnvironment(
        [ArmSpec("cheap", "A", 0.5, 4.0), ArmSpec("expensive", "B", 0.5, 20.0)],
        EnvironmentConfig(total_budget=5.0, max_rounds=2),
    )
    assert env.feasible_actions() == [0]
    assert env.group_opportunity_counts == {"A": 1, "B": 0}


def test_canonical_arm_environment_preserves_seeded_behavior() -> None:
    arms = [
        ArmSpec("A", "shared", 0.8, 12.0, 1.2),
        ArmSpec("B", "shared", 0.6, 8.0, 0.8),
        ArmSpec("C", "other", 0.4, 50.0),
    ]
    config = EnvironmentConfig(total_budget=40, max_rounds=3, seed=7)
    canonical = BanditEnvironment.for_arms(arms, config)
    legacy = ArmBanditEnvironment(arms, config)

    assert isinstance(canonical, BaseBudgetEnvironment)
    assert canonical.mode == "arm"
    assert canonical.n_actions == canonical.n_arms == 3
    assert canonical.current_contexts() is None
    assert canonical.current_costs() == pytest.approx(
        [12.001476184028979, 8.238996430006775, 50.0]
    )
    assert canonical.group_opportunity_counts == {"shared": 1, "other": 0}
    assert np.allclose(canonical.current_costs(), legacy.current_costs())

    for action in (0, 1, 0):
        canonical_result = canonical.step(action)
        legacy_result = legacy.step(action)
        assert canonical_result.reward == legacy_result.reward
        assert canonical_result.cost == legacy_result.cost
        assert canonical_result.observed_feedback == legacy_result.observed_feedback
        assert np.allclose(canonical.current_costs(), legacy.current_costs())

    assert canonical.done
    assert canonical.spent_budget <= config.total_budget + 1e-8


def test_named_constructor_rejects_wrong_configuration_type() -> None:
    arms = [ArmSpec("A", "A", 0.5, 5.0)]
    with pytest.raises(ValueError, match="At least one arm"):
        BanditEnvironment.for_arms(
            [], EnvironmentConfig(total_budget=10, max_rounds=2)
        )
    with pytest.raises(TypeError, match="arm mode requires EnvironmentConfig"):
        BanditEnvironment.for_arms(arms, contextual_config())
    with pytest.raises(TypeError, match="ContextualEnvironmentConfig"):
        BanditEnvironment.for_contextual(  # type: ignore[arg-type]
            EnvironmentConfig(total_budget=10, max_rounds=2)
        )


def test_legacy_aliases_and_common_action_validation() -> None:
    assert SyntheticEnvironment is ArmBanditEnvironment
    assert ContextualMicrofinanceEnv is ContextualBanditEnvironment

    env = BanditEnvironment.for_arms(
        [ArmSpec("A", "A", 0.5, 5.0)],
        EnvironmentConfig(total_budget=10, max_rounds=1),
    )
    with pytest.raises(IndexError, match="Invalid action"):
        env.step(-1)
    with pytest.raises(IndexError, match="Invalid action"):
        env.step(1)
    result = env.step(0)
    assert result.done
    with pytest.raises(RuntimeError, match="completed environment"):
        env.step(0)


def contextual_config(**overrides) -> ContextualEnvironmentConfig:
    values = {
        "total_budget": 100_000.0,
        "max_rounds": 5,
        "pool_size": 4,
        "seed": 3,
        "minimum_cost": 1.0,
    }
    values.update(overrides)
    return ContextualEnvironmentConfig(**values)


def test_contextual_reset_recreates_identical_candidate_stream() -> None:
    encoder = BorrowerFeatureEncoder()
    first = ContextualBanditEnvironment(contextual_config(), encoder)
    second = ContextualBanditEnvironment(contextual_config(), encoder)
    assert np.allclose(first.context_matrix(), second.context_matrix())
    assert np.allclose(first.current_costs(), second.current_costs())
    assert sum(first.group_opportunity_counts.values()) == len(first.feasible_actions())


def test_contextual_feedback_delay_and_pool_renewal() -> None:
    env = ContextualBanditEnvironment(
        contextual_config(feedback_delay_rounds=2), BorrowerFeatureEncoder()
    )
    first_ids = [candidate.borrower.id for candidate in env.current_pool]
    first_context = env.current_pool[env.feasible_actions()[0]].context.copy()
    first = env.step(env.feasible_actions()[0])
    assert not first.observed_feedback
    assert [candidate.borrower.id for candidate in env.current_pool] != first_ids
    env.step(env.feasible_actions()[0])
    third = env.step(env.feasible_actions()[0])
    assert len(third.observed_feedback) == 1
    np.testing.assert_array_equal(third.observed_feedback[0].context, first_context)
    assert env.spent_budget <= env.config.total_budget + 1e-8


def test_canonical_contextual_environment_matches_legacy_wrapper() -> None:
    config = contextual_config(feedback_delay_rounds=2, max_rounds=3)
    canonical = BanditEnvironment.for_contextual(config, BorrowerFeatureEncoder())
    legacy = ContextualBanditEnvironment(config, BorrowerFeatureEncoder())

    assert canonical.mode == "contextual"
    assert canonical.n_actions == config.pool_size
    assert canonical.n_features == canonical.context_matrix().shape[1]
    assert canonical.current_contexts() is not None
    assert np.allclose(canonical.context_matrix(), legacy.context_matrix())
    assert np.allclose(canonical.current_costs(), legacy.current_costs())

    for expected_feedback_count in (0, 0, 1):
        canonical_result = canonical.step(canonical.feasible_actions()[0])
        legacy_result = legacy.step(legacy.feasible_actions()[0])
        assert canonical_result.identifier == legacy_result.identifier
        assert canonical_result.reward == legacy_result.reward
        assert len(canonical_result.observed_feedback) == expected_feedback_count
        for canonical_feedback, legacy_feedback in zip(
            canonical_result.observed_feedback,
            legacy_result.observed_feedback,
        ):
            assert canonical_feedback.applicant_id == legacy_feedback.applicant_id
            assert canonical_feedback.reward == legacy_feedback.reward
            assert canonical_feedback.cost == legacy_feedback.cost
            assert np.allclose(canonical_feedback.context, legacy_feedback.context)
        assert np.allclose(canonical.current_costs(), legacy.current_costs())

    assert canonical.done
    assert len(canonical.pending_feedback) == 2
