import numpy as np
from Algorithms.bandit import BanditPolicy
from Environments.synthetic_environment_bandits import SyntheticEnvironment

class OracleRewardPerCostPolicy(BanditPolicy):
    """Evaluation-only policy that knows the true repayment probabilities."""

    def scores(self, env: SyntheticEnvironment) -> np.ndarray:
        probabilities = np.asarray(
            [arm.repayment_probability for arm in env.arms], dtype=float
        )
        return probabilities / env.current_costs
