from Algorithms.ucb import UCBPolicy
from Environments.base_environment import BanditEnvironment
import numpy as np

class BudgetedUCBPolicy(UCBPolicy):
    """UCB reward estimate divided by the current known loan cost."""

    def scores(self, env: BanditEnvironment) -> np.ndarray:
        """Compute the UCB reward estimate divided by the current known loan cost for each action."""
        reward_ucb = super().scores(env)
        return reward_ucb / env.current_costs()
