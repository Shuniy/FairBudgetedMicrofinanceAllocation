import numpy as np

from Algorithms.bandit import BanditPolicy
from Environments.base_environment import BanditEnvironment


class EpsilonFirstPolicy(BanditPolicy):
    """Spend a fixed budget fraction exploring, then exploit reward/cost."""

    def __init__(
        self,
        n_arms: int,
        exploration_budget_fraction: float = 0.2,
        seed: int = 0,
    ) -> None:
        """Initialize the Epsilon-First policy with a specified exploration budget fraction."""
        if not 0.0 < exploration_budget_fraction < 1.0:
            raise ValueError("exploration_budget_fraction must be between 0 and 1")
        self.exploration_budget_fraction = exploration_budget_fraction
        super().__init__(n_arms=n_arms, seed=seed)

    def scores(self, env: BanditEnvironment) -> np.ndarray:
        """Compute the scores for each arm based on the Epsilon-First strategy."""
        threshold = self.exploration_budget_fraction * env.config.total_budget
        if env.spent_budget < threshold:
            # Balanced exploration: prefer the least sampled arm.
            return -self.counts.astype(float)

        scores = np.full(self.n_arms, -np.inf, dtype=float)
        sampled = self.counts > 0
        scores[sampled] = self.mean_rewards[sampled] / env.current_costs()[sampled]
        if not np.any(sampled):
            return self.rng.random(self.n_arms)
        return scores
