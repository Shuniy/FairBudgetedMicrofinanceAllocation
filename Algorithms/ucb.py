import numpy as np

from Algorithms.bandit import BanditPolicy
from Environments.base_environment import BanditEnvironment


class UCBPolicy(BanditPolicy):
    """Upper Confidence Bound (UCB) policy for multi-armed bandit problems."""
    def __init__(
        self, n_arms: int, exploration_coefficient: float = 2.0, seed: int = 0
    ) -> None:
        if exploration_coefficient < 0:
            raise ValueError("exploration_coefficient cannot be negative")
        self.exploration_coefficient = exploration_coefficient
        super().__init__(n_arms=n_arms, seed=seed)

    def scores(self, env: BanditEnvironment) -> np.ndarray:
        """Compute the UCB scores for each arm based on observed rewards and counts."""
        scores = np.full(self.n_arms, np.inf, dtype=float)
        sampled = self.counts > 0
        if np.any(sampled):
            bonus = np.sqrt(
                self.exploration_coefficient
                * np.log(max(2, self.total_updates + 1))
                / self.counts[sampled]
            )
            scores[sampled] = self.mean_rewards[sampled] + bonus
        return scores
