import numpy as np

from Algorithms.bandit import BanditPolicy
from Environments.base_environment import BanditEnvironment
from models.events import FeedbackEvent


class ThompsonSamplingPolicy(BanditPolicy):
    """Thompson Sampling policy for multi-armed bandit problems."""
    def reset(self) -> None:
        super().reset()
        self.alpha = np.ones(self.n_arms, dtype=float)
        self.beta = np.ones(self.n_arms, dtype=float)

    def update(self, feedback: FeedbackEvent) -> None:
        """Update the policy based on the received feedback."""
        super().update(feedback)
        if feedback.action is None:
            raise ValueError("arm feedback must include an action")
        self.alpha[feedback.action] += feedback.reward
        self.beta[feedback.action] += 1.0 - feedback.reward

    def scores(self, env: BanditEnvironment) -> np.ndarray:
        """Sample from the posterior distribution for each arm to compute scores."""
        return self.rng.beta(self.alpha, self.beta)
