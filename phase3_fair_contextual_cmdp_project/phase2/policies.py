from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence

import numpy as np

from phase2.environment import ContextualMicrofinanceEnv
from phase2.metrics import allocation_shares, group_funding_rates


class ContextualPolicy(ABC):
    def __init__(self, n_features: int, seed: int = 0) -> None:
        if n_features <= 0:
            raise ValueError("n_features must be positive")
        self.n_features = n_features
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.total_updates = 0

    @abstractmethod
    def scores(self, env: ContextualMicrofinanceEnv) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def update(self, context: np.ndarray, reward: float, cost: float) -> None:
        raise NotImplementedError

    def select_action(
        self,
        env: ContextualMicrofinanceEnv,
        candidates: Sequence[int] | None = None,
    ) -> int:
        feasible = set(env.feasible_actions())
        options = list(feasible if candidates is None else candidates)
        options = [index for index in options if index in feasible]
        if not options:
            raise RuntimeError("No feasible borrower is available")
        scores = self.scores(env)
        best_value = max(scores[index] for index in options)
        best = [index for index in options if np.isclose(scores[index], best_value)]
        return int(self.rng.choice(best))


class RandomContextualPolicy(ContextualPolicy):
    def scores(self, env: ContextualMicrofinanceEnv) -> np.ndarray:
        return self.rng.random(len(env.current_pool))

    def update(self, context: np.ndarray, reward: float, cost: float) -> None:
        self.total_updates += 1


class LinearRidgeModel:
    def __init__(self, n_features: int, regularization: float = 1.0) -> None:
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
    def __init__(
        self,
        n_features: int,
        epsilon: float = 0.10,
        budget_aware: bool = False,
        regularization: float = 1.0,
        seed: int = 0,
    ) -> None:
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        self.epsilon = epsilon
        self.budget_aware = budget_aware
        self.regularization = regularization
        super().__init__(n_features, seed)

    def reset(self) -> None:
        super().reset()
        self.model = LinearRidgeModel(self.n_features, self.regularization)

    def scores(self, env: ContextualMicrofinanceEnv) -> np.ndarray:
        if self.rng.random() < self.epsilon:
            return self.rng.random(len(env.current_pool))
        predictions = np.clip(self.model.predict(env.context_matrix()), 0.0, 1.0)
        if self.budget_aware:
            predictions = predictions / env.current_costs()
        return predictions

    def update(self, context: np.ndarray, reward: float, cost: float) -> None:
        self.model.update(context, reward)
        self.total_updates += 1


class LinUCBPolicy(ContextualPolicy):
    def __init__(
        self,
        n_features: int,
        alpha: float = 1.0,
        budget_aware: bool = False,
        regularization: float = 1.0,
        seed: int = 0,
    ) -> None:
        if alpha < 0:
            raise ValueError("alpha cannot be negative")
        self.alpha = alpha
        self.budget_aware = budget_aware
        self.regularization = regularization
        super().__init__(n_features, seed)

    def reset(self) -> None:
        super().reset()
        self.model = LinearRidgeModel(self.n_features, self.regularization)

    def scores(self, env: ContextualMicrofinanceEnv) -> np.ndarray:
        contexts = env.context_matrix()
        inverse = np.linalg.inv(self.model.A)
        means = contexts @ self.model.theta
        uncertainty = np.sqrt(
            np.maximum(0.0, np.einsum("ij,jk,ik->i", contexts, inverse, contexts))
        )
        scores = means + self.alpha * uncertainty
        if self.budget_aware:
            scores = scores / env.current_costs()
        return scores

    def update(self, context: np.ndarray, reward: float, cost: float) -> None:
        self.model.update(context, reward)
        self.total_updates += 1


class ContextualThompsonSamplingPolicy(ContextualPolicy):
    """Gaussian linear Thompson Sampling approximation for binary rewards."""

    def __init__(
        self,
        n_features: int,
        posterior_scale: float = 0.35,
        budget_aware: bool = False,
        regularization: float = 1.0,
        seed: int = 0,
    ) -> None:
        if posterior_scale <= 0:
            raise ValueError("posterior_scale must be positive")
        self.posterior_scale = posterior_scale
        self.budget_aware = budget_aware
        self.regularization = regularization
        super().__init__(n_features, seed)

    def reset(self) -> None:
        super().reset()
        self.model = LinearRidgeModel(self.n_features, self.regularization)

    def scores(self, env: ContextualMicrofinanceEnv) -> np.ndarray:
        mean = self.model.theta
        covariance = (self.posterior_scale**2) * np.linalg.inv(self.model.A)
        sampled_theta = self.rng.multivariate_normal(mean, covariance)
        scores = env.context_matrix() @ sampled_theta
        if self.budget_aware:
            scores = scores / env.current_costs()
        return scores

    def update(self, context: np.ndarray, reward: float, cost: float) -> None:
        self.model.update(context, reward)
        self.total_updates += 1


class OracleContextualPolicy(ContextualPolicy):
    def __init__(self, n_features: int, budget_aware: bool = True, seed: int = 0) -> None:
        self.budget_aware = budget_aware
        super().__init__(n_features, seed)

    def scores(self, env: ContextualMicrofinanceEnv) -> np.ndarray:
        scores = env.true_probabilities()
        if self.budget_aware:
            scores = scores / env.current_costs()
        return scores

    def update(self, context: np.ndarray, reward: float, cost: float) -> None:
        self.total_updates += 1


class HardFairContextualPolicy(ContextualPolicy):
    """Quota first, demographic-parity correction second, utility third."""

    def __init__(
        self,
        base_policy: ContextualPolicy,
        target_shares: Mapping[str, float],
        parity_tolerance: float = 1.0,
        seed: int = 0,
    ) -> None:
        self.base_policy = base_policy
        self.target_shares = dict(target_shares)
        self.parity_tolerance = parity_tolerance
        super().__init__(base_policy.n_features, seed)

    def reset(self) -> None:
        if hasattr(self, "base_policy"):
            self.base_policy.reset()
        super().reset()

    def scores(self, env: ContextualMicrofinanceEnv) -> np.ndarray:
        return self.base_policy.scores(env)

    def update(self, context: np.ndarray, reward: float, cost: float) -> None:
        self.base_policy.update(context, reward, cost)
        self.total_updates = self.base_policy.total_updates

    def select_action(
        self,
        env: ContextualMicrofinanceEnv,
        candidates: Sequence[int] | None = None,
    ) -> int:
        feasible = list(env.feasible_actions() if candidates is None else candidates)
        groups = env.current_groups()

        t_next = env.total_allocations + 1
        deficits = {
            group: target * t_next - env.group_funding_counts.get(group, 0)
            for group, target in self.target_shares.items()
        }
        positive = sorted(
            (
                (group, value)
                for group, value in deficits.items()
                if value > 1e-12
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        for group, _ in positive:
            restricted = [index for index in feasible if groups[index] == group]
            if restricted:
                return self.base_policy.select_action(env, restricted)

        rates = group_funding_rates(
            env.group_funding_counts,
            env.group_opportunity_counts,
        )
        if rates and max(rates.values()) - min(rates.values()) > self.parity_tolerance:
            minimum = min(rates.values())
            disadvantaged = {
                group for group, rate in rates.items() if np.isclose(rate, minimum)
            }
            restricted = [index for index in feasible if groups[index] in disadvantaged]
            if restricted:
                return self.base_policy.select_action(env, restricted)

        return self.base_policy.select_action(env, feasible)


class SoftFairContextualPolicy(ContextualPolicy):
    def __init__(
        self,
        base_policy: ContextualPolicy,
        target_shares: Mapping[str, float],
        penalty_lambda: float,
        seed: int = 0,
    ) -> None:
        self.base_policy = base_policy
        self.target_shares = dict(target_shares)
        self.penalty_lambda = penalty_lambda
        super().__init__(base_policy.n_features, seed)

    def reset(self) -> None:
        if hasattr(self, "base_policy"):
            self.base_policy.reset()
        super().reset()

    def scores(self, env: ContextualMicrofinanceEnv) -> np.ndarray:
        scores = self.base_policy.scores(env).copy()
        shares = allocation_shares(env.group_funding_counts)
        groups = env.current_groups()
        for index, group in enumerate(groups):
            target = self.target_shares.get(group, 0.0)
            deficit = max(0.0, target - shares.get(group, 0.0))
            if np.isfinite(scores[index]):
                scores[index] += self.penalty_lambda * deficit
        return scores

    def update(self, context: np.ndarray, reward: float, cost: float) -> None:
        self.base_policy.update(context, reward, cost)
        self.total_updates = self.base_policy.total_updates
