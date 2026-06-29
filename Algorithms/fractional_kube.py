import numpy as np

from Algorithms.bandit import BanditPolicy
from Algorithms.ucb import UCBPolicy
from Environments.synthetic_environment_bandits import SyntheticEnvironment


class FractionalKUBEPolicy(UCBPolicy):
    """Fractional-knapsack approximation: optimistic reward/cost ratio."""

    def scores(self, env: SyntheticEnvironment) -> np.ndarray:
        optimistic_rewards = super().scores(env)
        return optimistic_rewards / env.current_costs


class OracleRewardPerCostPolicy(BanditPolicy):
    """Evaluation-only policy that knows the true repayment probabilities."""

    def scores(self, env: SyntheticEnvironment) -> np.ndarray:
        probabilities = np.asarray(
            [arm.repayment_probability for arm in env.arms], dtype=float
        )
        return probabilities / env.current_costs
