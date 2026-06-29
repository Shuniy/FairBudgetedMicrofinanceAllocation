import numpy as np

from Algorithms.bandit import BanditPolicy
from Algorithms.ucb import UCBPolicy
from Environments.synthetic_environment_bandits import SyntheticEnvironment


class ThompsonSamplingPolicy(BanditPolicy):
    def reset(self) -> None:
        super().reset()
        self.alpha = np.ones(self.n_arms, dtype=float)
        self.beta = np.ones(self.n_arms, dtype=float)

    def update(self, action: int, reward: float, cost: float) -> None:
        super().update(action, reward, cost)
        self.alpha[action] += reward
        self.beta[action] += 1.0 - reward

    def scores(self, env: SyntheticEnvironment) -> np.ndarray:
        return self.rng.beta(self.alpha, self.beta)


class RewardPerCostPolicy(BanditPolicy):
    """Greedy estimated repayment per unit of capital."""

    def scores(self, env: SyntheticEnvironment) -> np.ndarray:
        scores = np.full(self.n_arms, np.inf, dtype=float)
        sampled = self.counts > 0
        if np.any(sampled):
            # Current requested loan cost is known before funding.
            scores[sampled] = self.mean_rewards[sampled] / env.current_costs[sampled]
        return scores


class BudgetedUCBPolicy(UCBPolicy):
    """UCB reward estimate divided by the current known loan cost."""

    def scores(self, env: SyntheticEnvironment) -> np.ndarray:
        reward_ucb = super().scores(env)
        return reward_ucb / env.current_costs
