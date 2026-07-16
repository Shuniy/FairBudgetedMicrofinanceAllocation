from Algorithms.bandit import BanditPolicy
from Environments.base_environment import BanditEnvironment
import numpy as np

class RewardPerCostPolicy(BanditPolicy):
    """Greedy estimated repayment per unit of capital."""

    def scores(self, env: BanditEnvironment) -> np.ndarray:
        """Compute the estimated reward per cost for each action."""
        scores = np.full(self.n_arms, np.inf, dtype=float)
        sampled = self.counts > 0
        if np.any(sampled):
            # Current requested loan cost is known before funding.
            scores[sampled] = self.mean_rewards[sampled] / env.current_costs()[sampled]
        return scores
