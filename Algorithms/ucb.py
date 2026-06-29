import numpy as np

from Algorithms.bandit import BanditPolicy
from Environments.synthetic_environment_bandits import SyntheticEnvironment


class UCBPolicy(BanditPolicy):
    def __init__(
        self, n_arms: int, exploration_coefficient: float = 2.0, seed: int = 0
    ) -> None:
        if exploration_coefficient < 0:
            raise ValueError("exploration_coefficient cannot be negative")
        self.exploration_coefficient = exploration_coefficient
        super().__init__(n_arms=n_arms, seed=seed)

    def scores(self, env: SyntheticEnvironment) -> np.ndarray:
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
