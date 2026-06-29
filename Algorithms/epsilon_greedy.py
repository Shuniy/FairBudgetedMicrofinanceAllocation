import numpy as np

from Algorithms.bandit import BanditPolicy
from Environments.synthetic_environment_bandits import SyntheticEnvironment


class EpsilonGreedyPolicy(BanditPolicy):
    def __init__(
        self,
        n_arms: int,
        epsilon: float = 0.1,
        seed: int = 0,
        force_initial_exploration: bool = True,
    ) -> None:
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        self.epsilon = epsilon
        self.force_initial_exploration = force_initial_exploration
        super().__init__(n_arms=n_arms, seed=seed)

    def scores(self, env: SyntheticEnvironment) -> np.ndarray:
        if self.force_initial_exploration and np.any(self.counts == 0):
            return np.where(self.counts == 0, np.inf, -np.inf)
        if self.rng.random() < self.epsilon:
            return self.rng.random(self.n_arms)
        return self.mean_rewards.copy()


class DecayingEpsilonGreedyPolicy(EpsilonGreedyPolicy):
    def __init__(
        self,
        n_arms: int,
        initial_epsilon: float = 1.0,
        minimum_epsilon: float = 0.05,
        decay_rate: float = 0.02,
        seed: int = 0,
    ) -> None:
        if not 0 <= minimum_epsilon <= initial_epsilon <= 1:
            raise ValueError("Require 0 <= minimum_epsilon <= initial_epsilon <= 1")
        if decay_rate <= 0:
            raise ValueError("decay_rate must be positive")
        self.initial_epsilon = initial_epsilon
        self.minimum_epsilon = minimum_epsilon
        self.decay_rate = decay_rate
        super().__init__(n_arms=n_arms, epsilon=initial_epsilon, seed=seed)

    def scores(self, env: SyntheticEnvironment) -> np.ndarray:
        self.epsilon = max(
            self.minimum_epsilon,
            self.initial_epsilon / (1.0 + self.decay_rate * self.total_updates),
        )
        return super().scores(env)
