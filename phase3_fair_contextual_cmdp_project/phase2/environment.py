from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from phase2.feature_encoder import BorrowerFeatureEncoder
from phase2.models import (
    BorrowerCandidate,
    ContextualEnvironmentConfig,
    ContextualStepResult,
    FeedbackEvent,
)
from phase2.simulator import SyntheticBorrowerGenerator


class ContextualMicrofinanceEnv:
    """Dynamic borrower-pool environment for Phase 2.

    One round:
    1. a new pool of applicants is visible;
    2. the policy selects one affordable borrower;
    3. capital is deducted immediately;
    4. repayment is sampled and may be revealed after a configurable delay;
    5. the selected borrower leaves and a new pool arrives next round.
    """

    def __init__(
        self,
        config: ContextualEnvironmentConfig,
        encoder: BorrowerFeatureEncoder | None = None,
    ) -> None:
        self.config = config
        self.encoder = encoder or BorrowerFeatureEncoder()
        self.reset()

    @property
    def n_features(self) -> int:
        return self.encoder.n_features

    @property
    def done(self) -> bool:
        return self._done

    @property
    def total_allocations(self) -> int:
        return len(self.history)

    @property
    def spent_budget(self) -> float:
        return self.config.total_budget - self.remaining_budget

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        actual_seed = self.config.seed if seed is None else seed
        self.rng = np.random.default_rng(actual_seed)
        self.generator = SyntheticBorrowerGenerator(
            config=self.config,
            encoder=self.encoder,
            rng=self.rng,
        )
        self.remaining_budget = float(self.config.total_budget)
        self.round_index = 0
        self.history: list[dict[str, Any]] = []
        self.group_funding_counts: dict[str, int] = defaultdict(int)
        self.group_opportunity_counts: dict[str, int] = defaultdict(int)
        self.pending_feedback: list[FeedbackEvent] = []
        self.current_pool = self.generator.generate_pool(self.config.pool_size)
        self._record_pool_opportunities(self.current_pool)
        self._done = False
        self._refresh_done_flag()
        return self.observation()

    def _record_pool_opportunities(self, pool: list[BorrowerCandidate]) -> None:
        for candidate in pool:
            self.group_opportunity_counts[candidate.fairness_group] += 1
            # Ensure funded-count dictionaries also include zero-count groups.
            self.group_funding_counts.setdefault(candidate.fairness_group, 0)

    def feasible_actions(self) -> list[int]:
        if self.round_index >= self.config.max_rounds:
            return []
        return [
            index
            for index, candidate in enumerate(self.current_pool)
            if candidate.cost <= self.remaining_budget + 1e-12
        ]

    def context_matrix(self) -> np.ndarray:
        return np.vstack([candidate.context for candidate in self.current_pool])

    def current_costs(self) -> np.ndarray:
        return np.asarray([candidate.cost for candidate in self.current_pool], dtype=float)

    def current_groups(self) -> tuple[str, ...]:
        return tuple(candidate.fairness_group for candidate in self.current_pool)

    def true_probabilities(self) -> np.ndarray:
        return np.asarray(
            [candidate.true_repayment_probability for candidate in self.current_pool],
            dtype=float,
        )

    def observation(self) -> dict[str, Any]:
        return {
            "round_index": self.round_index,
            "remaining_budget": self.remaining_budget,
            "spent_budget": self.spent_budget,
            "contexts": self.context_matrix().copy(),
            "costs": self.current_costs().copy(),
            "groups": self.current_groups(),
            "feasible_actions": self.feasible_actions(),
            "group_funding_counts": dict(self.group_funding_counts),
            "group_opportunity_counts": dict(self.group_opportunity_counts),
        }

    def _release_due_feedback(self) -> tuple[FeedbackEvent, ...]:
        due = tuple(
            event for event in self.pending_feedback if event.due_round <= self.round_index
        )
        self.pending_feedback = [
            event for event in self.pending_feedback if event.due_round > self.round_index
        ]
        return due

    def step(self, action: int) -> ContextualStepResult:
        if self.done:
            raise RuntimeError("Cannot step a completed environment")
        if action < 0 or action >= len(self.current_pool):
            raise IndexError(f"Invalid action {action}")
        if action not in self.feasible_actions():
            raise ValueError("Selected borrower is not affordable")

        candidate = self.current_pool[action]
        reward = float(self.rng.binomial(1, candidate.true_repayment_probability))
        cost = float(candidate.cost)
        self.remaining_budget = max(0.0, self.remaining_budget - cost)
        self.group_funding_counts[candidate.fairness_group] += 1

        due_round = self.round_index + self.config.feedback_delay_rounds
        self.pending_feedback.append(
            FeedbackEvent(
                due_round=due_round,
                context=candidate.context.copy(),
                reward=reward,
                cost=cost,
                applicant_id=candidate.borrower.id,
            )
        )
        observed = self._release_due_feedback()

        funding_snapshot = dict(self.group_funding_counts)
        opportunity_snapshot = dict(self.group_opportunity_counts)
        row = {
            "round": self.round_index,
            "selected_index": action,
            "applicant_id": candidate.borrower.id,
            "fairness_group": candidate.fairness_group,
            "reward": reward,
            "cost": cost,
            "true_repayment_probability": candidate.true_repayment_probability,
            "budget_remaining": self.remaining_budget,
        }
        self.history.append(row)

        result_round = self.round_index
        self.round_index += 1
        if self.round_index < self.config.max_rounds:
            self.current_pool = self.generator.generate_pool(self.config.pool_size)
            self._record_pool_opportunities(self.current_pool)
        self._refresh_done_flag()

        return ContextualStepResult(
            round_index=result_round,
            selected_index=action,
            applicant_id=candidate.borrower.id,
            fairness_group=candidate.fairness_group,
            reward=reward,
            cost=cost,
            true_repayment_probability=candidate.true_repayment_probability,
            budget_remaining=self.remaining_budget,
            observed_feedback=observed,
            funding_counts=funding_snapshot,
            opportunity_counts=opportunity_snapshot,
            done=self.done,
        )

    def _refresh_done_flag(self) -> None:
        self._done = (
            self.round_index >= self.config.max_rounds
            or not self.feasible_actions()
        )
