from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np

from Environments.synthetic_environment_bandits import SyntheticEnvironment


class BanditPolicy(ABC):
    """Base class for bandit policies. Implements the basic functionality of tracking arm statistics and selecting actions based on scores."""

    def __init__(self, n_arms: int, seed: int = 0) -> None:
        if n_arms <= 0:
            raise ValueError("n_arms must be positive")
        self.n_arms = n_arms
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.counts = np.zeros(self.n_arms, dtype=int)
        self.reward_sums = np.zeros(self.n_arms, dtype=float)
        self.cost_sums = np.zeros(self.n_arms, dtype=float)
        self.total_updates = 0

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

    def update(self, action: int, reward: float, cost: float) -> None:
        self.counts[action] += 1
        self.reward_sums[action] += reward
        self.cost_sums[action] += cost
        self.total_updates += 1

    @abstractmethod
    def scores(self, env: SyntheticEnvironment) -> np.ndarray:
        """Return one action score per arm."""

    def select_action(
        self,
        env: SyntheticEnvironment,
        candidates: Sequence[int] | None = None,
    ) -> int:
        feasible = list(env.feasible_actions() if candidates is None else candidates)
        feasible = [action for action in feasible if action in env.feasible_actions()]
        if not feasible:
            raise RuntimeError("No feasible action available")

        action_scores = self.scores(env)
        feasible_scores = np.asarray([action_scores[a] for a in feasible], dtype=float)
        best_score = np.nanmax(feasible_scores)
        best = [
            action
            for action, score in zip(feasible, feasible_scores, strict=True)
            if np.isclose(score, best_score)
            or (np.isinf(score) and np.isinf(best_score))
        ]
        return int(self.rng.choice(best))


class RandomPolicy(BanditPolicy):
    def scores(self, env: SyntheticEnvironment):
        return self.rng.random(self.n_arms)
