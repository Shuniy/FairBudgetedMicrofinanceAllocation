import numpy as np

from Algorithms.bandit import BanditPolicy
from Algorithms.ucb import UCBPolicy
from Environments.synthetic_environment_bandits import SyntheticEnvironment


class FractionalKUBEPolicy(UCBPolicy):
    """Fractional-knapsack approximation: optimistic reward/cost ratio."""

    def scores(self, env: SyntheticEnvironment) -> np.ndarray:
        optimistic_rewards = super().scores(env)
        return optimistic_rewards / env.current_costs
