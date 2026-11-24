from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from enum import Enum
from typing import Any

import numpy as np

from Data.feature_encoder import BorrowerFeatureEncoder
from models.arm_spec import ArmSpec
from models.borrower_candidate import BorrowerCandidate
from models.environment_config import ContextualEnvironmentConfig, EnvironmentConfig
from models.events import FeedbackEvent, StepResult
from simulation import SyntheticBorrowerGenerator


class BanditMode(str, Enum):
    ARM = "arm"
    CONTEXTUAL = "contextual"

class BanditEnvironment:
    """A bandit environment for either fixed-arm or contextual borrower-pool settings."""
    def __init__(
        self,
        *,
        mode: BanditMode,
        config: EnvironmentConfig,
        arms: Sequence[ArmSpec] | None = None,
        encoder: BorrowerFeatureEncoder | None = None,
    ) -> None:
        """Initialize the bandit environment."""
        if mode == BanditMode.ARM:
            if isinstance(config, ContextualEnvironmentConfig):
                raise TypeError("arm mode requires EnvironmentConfig")
            if not arms:
                raise ValueError("At least one arm is required")
            self.arms = tuple(arms)
        elif mode == "contextual":
            if not isinstance(config, ContextualEnvironmentConfig):
                raise TypeError(
                    "contextual mode requires ContextualEnvironmentConfig"
                )
            if arms is not None:
                raise TypeError("contextual mode does not accept arms")
            self.encoder = encoder or BorrowerFeatureEncoder()
        else:
            raise ValueError(f"Unsupported bandit mode: {mode}")

        self.mode: BanditMode = mode
        self.config = config
        self.remaining_budget = float(config.total_budget)
        self.round_index = 0
        self.history: list[dict[str, Any]] = []
        self.group_funding_counts: dict[str, int] = {}
        self.group_opportunity_counts: dict[str, int] = {}
        self.group_population_opportunity_counts: dict[str, int] = {}
        self.pending_feedback: list[FeedbackEvent] = []
        self._done = False
        self.reset()

    @staticmethod
    def for_arms(
        arms: Sequence[ArmSpec],
        config: EnvironmentConfig,
    ) -> BanditEnvironment:
        """Create a fixed-arm environment."""

        return BanditEnvironment(mode=BanditMode.ARM, arms=arms, config=config)

    @staticmethod
    def for_contextual(
        config: ContextualEnvironmentConfig,
        encoder: BorrowerFeatureEncoder | None = None,
    ) -> BanditEnvironment:
        """Create a contextual borrower environment."""

        return BanditEnvironment(mode=BanditMode.CONTEXTUAL, config=config, encoder=encoder)

    @property
    def contextual_config(self) -> ContextualEnvironmentConfig:
        """Get the contextual environment configuration."""
        if self.mode != "contextual":
            raise TypeError("contextual_config is only available in contextual mode")
        return self.config  # type: ignore[return-value]

    @property
    def spent_budget(self) -> float:
        """Get the total budget spent so far."""
        return self.config.total_budget - self.remaining_budget

    @property
    def total_allocations(self) -> int:
        """Get the total number of allocations made so far."""
        return len(self.history)

    @property
    def done(self) -> bool:
        return self._done

    @property
    def n_actions(self) -> int:
        return len(self.current_costs())

    @property
    def n_arms(self) -> int:
        if self.mode != BanditMode.ARM:
            raise TypeError("n_arms is only available in arm mode")
        return len(self.arms)

    @property
    def n_features(self) -> int:
        """Get the number of features in the borrower context."""
        if self.mode != "contextual":
            raise TypeError("n_features is only available in contextual mode")
        return self.encoder.n_features

    @property
    def groups(self) -> tuple[str, ...]:
        """Get the unique fairness groups present in the environment."""
        if self.mode == BanditMode.ARM:
            return tuple(dict.fromkeys(arm.group for arm in self.arms))
        return tuple(dict.fromkeys(self.current_groups()))

    @property
    def group_allocation_counts(self) -> dict[str, int]:
        return self.group_funding_counts

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        """Reset the environment to its initial state."""
        actual_seed = self.config.seed if seed is None else seed
        self.rng = np.random.default_rng(actual_seed)
        self.remaining_budget = float(self.config.total_budget)
        self.round_index = 0
        self.history = []
        self.pending_feedback = []
        self.group_population_opportunity_counts = defaultdict(int)

        if self.mode == BanditMode.ARM:
            self.arm_allocation_counts = np.zeros(self.n_arms, dtype=int)
            self.group_funding_counts = defaultdict(
                int, {group: 0 for group in self.groups}
            )
            self.group_opportunity_counts = defaultdict(
                int, {group: 0 for group in self.groups}
            )
            self._current_costs = self._sample_arm_costs()
        else:
            self.group_funding_counts = defaultdict(int)
            self.group_opportunity_counts = defaultdict(int)
            self.generator = SyntheticBorrowerGenerator(
                config=self.contextual_config,
                encoder=self.encoder,
                rng=self.rng,
            )
            self.current_pool = self.generator.generate_pool(
                self.contextual_config.pool_size
            )

        self._done = False
        self._record_current_opportunities()
        self._refresh_done_flag()
        return self.observation()

    def _sample_arm_costs(self) -> np.ndarray:
        """Sample costs for each arm based on their mean and standard deviation."""
        costs = []
        for arm in self.arms:
            sampled = (
                arm.cost_mean
                if arm.cost_std == 0
                else self.rng.normal(arm.cost_mean, arm.cost_std)
            )
            costs.append(max(self.config.minimum_cost, float(sampled)))
        return np.asarray(costs, dtype=float)

    def current_costs(self) -> np.ndarray:
        """Get the current costs for each action (arm or borrower)."""
        if self.mode == BanditMode.ARM:
            return self._current_costs
        return np.asarray(
            [candidate.cost for candidate in self.current_pool], dtype=float
        )

    def current_groups(self) -> tuple[str, ...]:
        """Get the fairness groups for the current actions (arms or borrowers)."""
        if self.mode == BanditMode.ARM:
            return tuple(arm.group for arm in self.arms)
        return tuple(candidate.fairness_group for candidate in self.current_pool)

    def current_contexts(self) -> np.ndarray | None:
        """Get the context matrix for the current borrower pool. Returns None in arm mode."""
        if self.mode == BanditMode.ARM:
            return None
        return np.vstack([candidate.context for candidate in self.current_pool])

    def context_matrix(self) -> np.ndarray:
        """Get the context matrix for the current borrower pool. Raises an error in arm mode."""
        contexts = self.current_contexts()
        if contexts is None:
            raise TypeError("context_matrix is only available in contextual mode")
        return contexts

    def true_probabilities(self) -> np.ndarray:
        """Get the true repayment probabilities for the current actions (arms or borrowers)."""
        if self.mode == BanditMode.ARM:
            return np.asarray(
                [arm.repayment_probability for arm in self.arms], dtype=float
            )
        return np.asarray(
            [
                candidate.true_repayment_probability
                for candidate in self.current_pool
            ],
            dtype=float,
        )

    def feasible_actions(self) -> list[int]:
        """Get the indices of actions that are feasible given the remaining budget."""
        if self.round_index >= self.config.max_rounds:
            return []
        return [
            index
            for index, cost in enumerate(self.current_costs())
            if cost <= self.remaining_budget + 1e-12
        ]

    def _record_current_opportunities(self) -> None:
        """Count arrivals once, with a separate count of affordable opportunities."""
        if self.mode == BanditMode.ARM:
            for group in {self.arms[index].group for index in self.feasible_actions()}:
                self.group_opportunity_counts[group] += 1
            return

        groups = self.current_groups()
        for group in groups:
            self.group_population_opportunity_counts[group] += 1
        for index in self.feasible_actions():
            group = groups[index]
            self.group_opportunity_counts[group] += 1
            self.group_funding_counts.setdefault(group, 0)

    def observation(self) -> dict[str, Any]:
        """Get the current observation of the environment, including round index, budget, costs, groups, and feasible actions."""
        observation: dict[str, Any] = {
            "round_index": self.round_index,
            "remaining_budget": self.remaining_budget,
            "spent_budget": self.spent_budget,
            "costs": self.current_costs().copy(),
            "groups": self.current_groups(),
            "feasible_actions": self.feasible_actions(),
            "group_funding_counts": dict(self.group_funding_counts),
            "group_opportunity_counts": dict(self.group_opportunity_counts),
        }
        if self.mode == BanditMode.ARM:
            observation["arm_allocation_counts"] = self.arm_allocation_counts.copy()
        else:
            observation["contexts"] = self.context_matrix().copy()
            observation["group_population_opportunity_counts"] = dict(
                self.group_population_opportunity_counts
            )
        return observation

    def _release_due_feedback(self) -> tuple[FeedbackEvent, ...]:
        """Release feedback events that are due in the current round and remove them from the pending feedback list."""
        due = tuple(
            event
            for event in self.pending_feedback
            if event.due_round <= self.round_index
        )
        self.pending_feedback = [
            event
            for event in self.pending_feedback
            if event.due_round > self.round_index
        ]
        return due

    def _refresh_done_flag(self) -> None:
        self._done = (
            self.round_index >= self.config.max_rounds
            or not self.feasible_actions()
        )

    def _selected_choice(
        self, action: int
    ) -> tuple[str, str, float, np.ndarray | None, dict[str, Any]]:
        """Get the details of the selected action, including identifier, group, repayment probability, context, and metadata."""
        if self.mode == BanditMode.ARM:
            arm = self.arms[action]
            return (
                arm.name,
                arm.group,
                arm.repayment_probability,
                None,
                {"arm_name": arm.name},
            )

        candidate: BorrowerCandidate = self.current_pool[action]
        return (
            candidate.borrower.id,
            candidate.fairness_group,
            candidate.true_repayment_probability,
            candidate.context.copy(),
            {
                "applicant_id": candidate.borrower.id,
                "poverty_relief_score": candidate.borrower.poverty_relief_score,
                "vulnerability_score": candidate.borrower.vulnerability_score,
            },
        )

    def step(self, action: int) -> StepResult:
        """Take a step in the environment by selecting an action (arm or borrower) and return the result."""
        if self.done:
            raise RuntimeError("Cannot step a completed environment")
        if action < 0 or action >= self.n_actions:
            raise IndexError(f"Invalid action {action}")
        if action not in self.feasible_actions():
            if self.mode == BanditMode.ARM:
                raise ValueError("Selected arm is not affordable")
            raise ValueError("Selected borrower is not affordable")

        identifier, group, probability, context, metadata = self._selected_choice(
            action
        )
        cost = float(self.current_costs()[action])
        reward = float(self.rng.binomial(1, probability))
        self.remaining_budget = max(0.0, self.remaining_budget - cost)
        if self.mode == BanditMode.ARM:
            self.arm_allocation_counts[action] += 1
        self.group_funding_counts[group] += 1
        funding_snapshot = dict(self.group_funding_counts)
        opportunity_snapshot = dict(self.group_opportunity_counts)
        population_snapshot = dict(self.group_population_opportunity_counts)

        if self.mode == BanditMode.ARM:
            observed = (
                FeedbackEvent(
                    due_round=self.round_index,
                    action=action,
                    reward=reward,
                    cost=cost,
                ),
            )
        else:
            self.pending_feedback.append(
                FeedbackEvent(
                    due_round=(
                        self.round_index
                        + self.contextual_config.feedback_delay_rounds
                    ),
                    action=action,
                    context=context,
                    reward=reward,
                    cost=cost,
                    applicant_id=identifier,
                )
            )
            observed = self._release_due_feedback()

        result_round = self.round_index
        self.history.append(
            {
                "round": result_round,
                "selected_index": action,
                "identifier": identifier,
                "group": group,
                "reward": reward,
                "cost": cost,
            }
        )
        self.round_index += 1
        if self.mode == BanditMode.ARM:
            self._current_costs = self._sample_arm_costs()
            self._record_current_opportunities()
        elif self.round_index < self.config.max_rounds:
            self.current_pool = self.generator.generate_pool(
                self.contextual_config.pool_size
            )
            self._record_current_opportunities()
        self._refresh_done_flag()

        return StepResult(
            round_index=result_round,
            selected_index=action,
            identifier=identifier,
            group=group,
            reward=reward,
            cost=cost,
            budget_remaining=self.remaining_budget,
            observed_feedback=observed,
            funding_counts=funding_snapshot,
            opportunity_counts=opportunity_snapshot,
            population_opportunity_counts=population_snapshot,
            true_repayment_probability=probability,
            metadata=metadata,
            done=self.done,
        )


# Compatibility name retained for existing type annotations and imports.
BaseBudgetEnvironment = BanditEnvironment
