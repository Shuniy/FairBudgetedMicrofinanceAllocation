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
