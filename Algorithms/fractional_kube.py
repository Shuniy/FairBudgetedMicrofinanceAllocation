import numpy as np

from Algorithms.ucb import UCBPolicy
from Environments.base_environment import BanditEnvironment


class FractionalKUBEPolicy(UCBPolicy):
    """Fractional-knapsack approximation: optimistic reward/cost ratio."""

    def scores(self, env: BanditEnvironment) -> np.ndarray:
        optimistic_rewards = super().scores(env)
        return optimistic_rewards / env.current_costs()
