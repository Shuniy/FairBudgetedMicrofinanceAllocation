from __future__ import annotations
import numpy as np

from Algorithms.bandit import BanditPolicy
from Environments.base_environment import BanditEnvironment

class TrueRewardPerCostPolicy(BanditPolicy):
    """Evaluation-only policy that knows the true repayment probabilities."""

    def scores(self, env: BanditEnvironment) -> np.ndarray:
        """Compute the true reward per cost for each action."""
        return env.true_probabilities() / env.current_costs()
