from __future__ import annotations

import numpy as np

from Algorithms.bandit import BasePolicy
from Environments.base_environment import BanditEnvironment
from models.events import FeedbackEvent


def _contextual_environment(env: BanditEnvironment) -> BanditEnvironment:
    if env.mode != "contextual":
        raise TypeError("contextual policy requires a contextual environment")
    return env


class ContextualPolicy(BasePolicy):
    """Base class for contextual bandit policies."""
    def __init__(self, n_features: int, seed: int = 0) -> None:
        """Initialize the ContextualPolicy with the number of features and a random seed."""
        if n_features <= 0:
            raise ValueError("n_features must be positive")
        self.n_features = n_features
        super().__init__(seed)


class RandomContextualPolicy(ContextualPolicy):
    """Random policy for contextual bandit problems."""
    def scores(self, env: BanditEnvironment) -> np.ndarray:
        return self.rng.random(_contextual_environment(env).n_actions)

    def update(self, feedback: FeedbackEvent) -> None:
        """Update the policy based on the received feedback. For the random policy, no updates are needed."""
        self.total_updates += 1


class LinearRidgeModel:
    """Linear ridge regression model for contextual bandit policies."""
    def __init__(self, n_features: int, regularization: float = 1.0) -> None:
        """Initialize the LinearRidgeModel with the number of features and regularization parameter."""
        if n_features <= 0 or regularization <= 0:
            raise ValueError("n_features and regularization must be positive")
        self.n_features = n_features
        self.regularization = regularization
        self.reset()

    def reset(self) -> None:
        self.A = np.eye(self.n_features) * self.regularization
        self.b = np.zeros(self.n_features)

    @property
    def theta(self) -> np.ndarray:
        return np.linalg.solve(self.A, self.b)

    def predict(self, contexts: np.ndarray) -> np.ndarray:
        return contexts @ self.theta

    def update(self, context: np.ndarray, reward: float) -> None:
        self.A += np.outer(context, context)
        self.b += reward * context


class ContextualEpsilonGreedyPolicy(ContextualPolicy):
    """Epsilon-Greedy policy for contextual bandit problems."""
    def __init__(
        self,
        n_features: int,
        epsilon: float = 0.10,
        budget_aware: bool = False,
        regularization: float = 1.0,
        seed: int = 0,
    ) -> None:
        """Initialize the ContextualEpsilonGreedyPolicy with the number of features, epsilon value, budget awareness flag, and regularization parameter."""
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        self.epsilon = epsilon
        self.budget_aware = budget_aware
        self.regularization = regularization
        super().__init__(n_features, seed)

    def reset(self) -> None:
        super().reset()
        self.model = LinearRidgeModel(self.n_features, self.regularization)

    def scores(self, env: BanditEnvironment) -> np.ndarray:
        """Compute the scores for each action based on the epsilon-greedy strategy in a contextual bandit setting."""
        contextual = _contextual_environment(env)
        if self.rng.random() < self.epsilon:
            return self.rng.random(contextual.n_actions)
        predictions = np.clip(
            self.model.predict(contextual.context_matrix()), 0.0, 1.0
        )
        return (
            predictions / contextual.current_costs()
            if self.budget_aware
            else predictions
        )

    def update(self, feedback: FeedbackEvent) -> None:
        """Update the policy based on the received feedback. For the epsilon-greedy policy, update the linear model with the context and reward."""
        if feedback.context is None:
            raise ValueError("contextual feedback must include a context")
        self.model.update(feedback.context, feedback.reward)
        self.total_updates += 1


class LinUCBPolicy(ContextualPolicy):
    """LinUCB policy for contextual bandit problems."""
    def __init__(
        self,
        n_features: int,
        alpha: float = 1.0,
        budget_aware: bool = False,
        regularization: float = 1.0,
        seed: int = 0,
    ) -> None:
        """Initialize the LinUCBPolicy with the number of features, alpha value, budget awareness flag, and regularization parameter."""
        if alpha < 0:
            raise ValueError("alpha cannot be negative")
        self.alpha = alpha
        self.budget_aware = budget_aware
        self.regularization = regularization
        super().__init__(n_features, seed)

    def reset(self) -> None:
        super().reset()
        self.model = LinearRidgeModel(self.n_features, self.regularization)

    def scores(self, env: BanditEnvironment) -> np.ndarray:
        """Compute the scores for each action based on the LinUCB strategy in a contextual bandit setting."""
        contextual = _contextual_environment(env)
        contexts = contextual.context_matrix()
        inverse = np.linalg.inv(self.model.A)
        means = contexts @ self.model.theta
        uncertainty = np.sqrt(
            np.maximum(0.0, np.einsum("ij,jk,ik->i", contexts, inverse, contexts))
        )
        scores = means + self.alpha * uncertainty
        return scores / contextual.current_costs() if self.budget_aware else scores

    def update(self, feedback: FeedbackEvent) -> None:
        """Update the policy based on the received feedback. For the LinUCB policy, update the linear model with the context and reward."""
        if feedback.context is None:
            raise ValueError("contextual feedback must include a context")
        self.model.update(feedback.context, feedback.reward)
        self.total_updates += 1


class ContextualThompsonSamplingPolicy(ContextualPolicy):
    """Contextual Thompson Sampling policy for contextual bandit problems."""
    def __init__(
        self,
        n_features: int,
        posterior_scale: float = 0.35,
        budget_aware: bool = False,
        regularization: float = 1.0,
        seed: int = 0,
    ) -> None:
        """Initialize the ContextualThompsonSamplingPolicy with the number of features, posterior scale, budget awareness flag, and regularization parameter."""
        if posterior_scale <= 0:
            raise ValueError("posterior_scale must be positive")
        self.posterior_scale = posterior_scale
        self.budget_aware = budget_aware
        self.regularization = regularization
        super().__init__(n_features, seed)

    def reset(self) -> None:
        super().reset()
        self.model = LinearRidgeModel(self.n_features, self.regularization)

    def scores(self, env: BanditEnvironment) -> np.ndarray:
        """Compute the scores for each action based on the Contextual Thompson Sampling strategy."""
        contextual = _contextual_environment(env)
        covariance = (self.posterior_scale**2) * np.linalg.inv(self.model.A)
        sampled_theta = self.rng.multivariate_normal(self.model.theta, covariance)
        scores = contextual.context_matrix() @ sampled_theta
        return scores / contextual.current_costs() if self.budget_aware else scores

    def update(self, feedback: FeedbackEvent) -> None:
        """Update the policy based on the received feedback. For the Contextual Thompson Sampling policy, update the linear model with the context and reward."""
        if feedback.context is None:
            raise ValueError("contextual feedback must include a context")
        self.model.update(feedback.context, feedback.reward)
        self.total_updates += 1


class TrueRewardContextualPolicy(ContextualPolicy):
    """Informed greedy benchmark using true repayment, per cost by default."""

    def __init__(
        self, n_features: int, budget_aware: bool = True, seed: int = 0
    ) -> None:
        """Initialize the TrueRewardContextualPolicy with the number of features, budget awareness flag, and random seed."""
        self.budget_aware = budget_aware
        super().__init__(n_features, seed)

    def scores(self, env: BanditEnvironment) -> np.ndarray:
        """Compute the scores for each action based on the true repayment probabilities."""
        contextual = _contextual_environment(env)
        scores = contextual.true_probabilities()
        return scores / contextual.current_costs() if self.budget_aware else scores

    def update(self, feedback: FeedbackEvent) -> None:
        self.total_updates += 1
