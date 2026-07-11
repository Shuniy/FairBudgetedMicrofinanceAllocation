from Algorithms.bandit import BanditPolicy
from Environments.synthetic_environment_bandits import SyntheticEnvironment
import numpy as np

class RewardPerCostPolicy(BanditPolicy):
    """Greedy estimated repayment per unit of capital."""

    def scores(self, env: SyntheticEnvironment) -> np.ndarray:
        scores = np.full(self.n_arms, np.inf, dtype=float)
        sampled = self.counts > 0
        if np.any(sampled):
            # Current requested loan cost is known before funding.
            scores[sampled] = self.mean_rewards[sampled] / env.current_costs[sampled]
        return scores
