from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np

from Environments.base_environment import BanditEnvironment
from models.events import FeedbackEvent


class BasePolicy(ABC):
    """Shared policy lifecycle and feasible score-based action selection."""

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self.reset()

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.total_updates = 0

    @abstractmethod
    def scores(self, env: BanditEnvironment) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def update(self, feedback: FeedbackEvent) -> None:
        raise NotImplementedError

    def _scores_for_options(
        self, env: BanditEnvironment, options: list[int]
    ) -> np.ndarray:
        """Compute selection scores once the eligible options are known."""
        return self.scores(env)

    def select_action(
        self,
        env: BanditEnvironment,
        candidates: Sequence[int] | None = None,
    ) -> int:
        """Select an action based on the policy's scores and feasibility constraints."""
        feasible = set(env.feasible_actions())
        options = sorted(feasible if candidates is None else candidates)
        options = [index for index in options if index in feasible]
        if not options:
            raise RuntimeError("No feasible action available")
        action_scores = self._scores_for_options(env, options)
        best_value = max(action_scores[index] for index in options)
        best = [
            index
            for index in options
            if np.isclose(action_scores[index], best_value)
        ]
        return int(self.rng.choice(best))


class ArmPolicy(BasePolicy):
    """Base class for multi-armed bandit policies."""
    def __init__(self, n_arms: int, seed: int = 0) -> None:
        """Initialize the ArmPolicy with the number of arms and a random seed."""
        if n_arms <= 0:
            raise ValueError("n_arms must be positive")
        self.n_arms = n_arms
        super().__init__(seed)

    def reset(self) -> None:
        super().reset()
        self.counts = np.zeros(self.n_arms, dtype=int)
        self.reward_sums = np.zeros(self.n_arms, dtype=float)
        self.cost_sums = np.zeros(self.n_arms, dtype=float)

    @property
    def mean_rewards(self) -> np.ndarray:
        return np.divide(
            self.reward_sums,
            self.counts,
            out=np.zeros_like(self.reward_sums),
            where=self.counts > 0,
        )

    @property
    def mean_costs(self) -> np.ndarray:
        return np.divide(
            self.cost_sums,
            self.counts,
            out=np.full_like(self.cost_sums, np.inf),
            where=self.counts > 0,
        )

    def update(self, feedback: FeedbackEvent) -> None:
        """Update the policy based on the received feedback."""
        if feedback.action is None:
            raise ValueError("arm feedback must include an action")
        action = feedback.action
        self.counts[action] += 1
        self.reward_sums[action] += feedback.reward
        self.cost_sums[action] += feedback.cost
        self.total_updates += 1


class RandomPolicy(ArmPolicy):
    """Random policy for multi-armed bandit problems."""
    def scores(self, env: BanditEnvironment) -> np.ndarray:
        """Compute random scores for each arm."""
        return self.rng.random(self.n_arms)


# Compatibility name retained for external imports.
BanditPolicy = ArmPolicy
