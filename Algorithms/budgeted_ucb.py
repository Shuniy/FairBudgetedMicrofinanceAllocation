from Algorithms.ucb import UCBPolicy
from Environments.synthetic_environment_bandits import SyntheticEnvironment
import numpy as np

class BudgetedUCBPolicy(UCBPolicy):
    """UCB reward estimate divided by the current known loan cost."""

    def scores(self, env: SyntheticEnvironment) -> np.ndarray:
        reward_ucb = super().scores(env)
        return reward_ucb / env.current_costs
