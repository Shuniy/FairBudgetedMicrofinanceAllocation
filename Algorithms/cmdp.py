from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence

import numpy as np

from Algorithms.contextual import LinearRidgeModel
from Environments.cmdp_environment import StatefulMicrofinanceCMDPEnv
from metrics import demographic_parity_gap, total_quota_deficit
from models.fairness_config import LagrangianConfig


class CMDPPolicy(ABC):
    """Policy lifecycle for the stateful Phase 3 environment."""

    def __init__(self, n_features: int, seed: int = 0) -> None:
        if n_features <= 0:
            raise ValueError("n_features must be positive")
        self.n_features = n_features
        self.seed = seed
        self.reset()

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.total_updates = 0

    @abstractmethod
    def scores(self, env: StatefulMicrofinanceCMDPEnv) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def decision_context(
        self,
        env: StatefulMicrofinanceCMDPEnv,
        action: int,
    ) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def update(self, context: np.ndarray, reward: float, cost: float) -> None:
        raise NotImplementedError

    def on_round_end(self, env: StatefulMicrofinanceCMDPEnv) -> None:
        """Optional hook for state or dual-variable updates."""

    def select_action(
        self,
        env: StatefulMicrofinanceCMDPEnv,
        candidates: Sequence[int] | None = None,
    ) -> int | None:
        feasible = set(env.feasible_actions())
        options = sorted(feasible if candidates is None else candidates)
        options = [index for index in options if index in feasible]
        if not options:
            return None
        scores = self.scores(env)
        best_value = max(scores[index] for index in options)
        best = [
            index
            for index in options
            if np.isclose(scores[index], best_value)
            or (np.isinf(scores[index]) and np.isinf(best_value))
        ]
        return int(self.rng.choice(best))


class RandomCMDPPolicy(CMDPPolicy):
    def scores(self, env: StatefulMicrofinanceCMDPEnv) -> np.ndarray:
        return self.rng.random(len(env.current_pool))

    def decision_context(
        self, env: StatefulMicrofinanceCMDPEnv, action: int
    ) -> np.ndarray:
        return env.current_pool[action].context.copy()

    def update(self, context: np.ndarray, reward: float, cost: float) -> None:
        self.total_updates += 1


class ContextOnlyBudgetedLinUCBPolicy(CMDPPolicy):
    """Phase 2-style control using context and cost but no portfolio state."""

    def __init__(
        self,
        n_features: int,
        alpha: float = 0.70,
        regularization: float = 1.0,
        seed: int = 0,
    ) -> None:
        if alpha < 0:
            raise ValueError("alpha cannot be negative")
        self.alpha = alpha
        self.regularization = regularization
        super().__init__(n_features, seed)

    def reset(self) -> None:
        super().reset()
        self.model = LinearRidgeModel(self.n_features, self.regularization)

    def _optimistic_reward(
        self, env: StatefulMicrofinanceCMDPEnv
    ) -> np.ndarray:
        contexts = env.base_context_matrix()
        if len(contexts) == 0:
            return np.array([], dtype=float)
        inverse = np.linalg.inv(self.model.A)
        means = contexts @ self.model.theta
        uncertainty = np.sqrt(
            np.maximum(
                0.0,
                np.einsum("ij,jk,ik->i", contexts, inverse, contexts),
            )
        )
        return means + self.alpha * uncertainty

    def scores(self, env: StatefulMicrofinanceCMDPEnv) -> np.ndarray:
        optimistic = self._optimistic_reward(env)
        if len(optimistic) == 0:
            return optimistic
        costs = env.current_costs()
        median_cost = max(float(np.median(costs)), 1.0)
        normalized_cost = np.maximum(costs / median_cost, 1e-6)
        return optimistic / normalized_cost

    def decision_context(
        self, env: StatefulMicrofinanceCMDPEnv, action: int
    ) -> np.ndarray:
        return env.current_pool[action].context.copy()

    def update(self, context: np.ndarray, reward: float, cost: float) -> None:
        self.model.update(context, reward)
        self.total_updates += 1


class PortfolioAwareLinUCBPolicy(ContextOnlyBudgetedLinUCBPolicy):
    """LinUCB with a state-dependent shadow price for liquid capital."""

    def __init__(
        self,
        n_features: int,
        alpha: float = 0.70,
        base_shadow_price: float = 0.22,
        liquidity_weight: float = 1.40,
        due_soon_relief: float = 0.80,
        default_loss_weight: float = 0.60,
        regularization: float = 1.0,
        seed: int = 0,
    ) -> None:
        if min(
            base_shadow_price,
            liquidity_weight,
            due_soon_relief,
            default_loss_weight,
        ) < 0:
            raise ValueError("portfolio shadow-price parameters cannot be negative")
        self.base_shadow_price = base_shadow_price
        self.liquidity_weight = liquidity_weight
        self.due_soon_relief = due_soon_relief
        self.default_loss_weight = default_loss_weight
        super().__init__(
            n_features=n_features,
            alpha=alpha,
            regularization=regularization,
            seed=seed,
        )

    def capital_shadow_price(self, env: StatefulMicrofinanceCMDPEnv) -> float:
        initial = env.config.initial_capital
        liquidity_ratio = np.clip(env.available_capital / initial, 0.0, 1.5)
        liquidity_pressure = max(0.0, 1.0 - liquidity_ratio)
        due_soon_ratio = np.clip(env.due_soon_capital(2) / initial, 0.0, 1.0)
        default_loss_ratio = np.clip(
            env.cumulative_default_loss / initial, 0.0, 1.0
        )
        numerator = self.base_shadow_price * (
            1.0
            + self.liquidity_weight * liquidity_pressure
            + self.default_loss_weight * default_loss_ratio
        )
        return float(
            numerator / (1.0 + self.due_soon_relief * due_soon_ratio)
        )

    def scores(self, env: StatefulMicrofinanceCMDPEnv) -> np.ndarray:
        optimistic = self._optimistic_reward(env)
        if len(optimistic) == 0:
            return optimistic
        costs = env.current_costs()
        median_cost = max(float(np.median(costs)), 1.0)
        return optimistic - self.capital_shadow_price(env) * (costs / median_cost)


class PortfolioAwareThompsonPolicy(CMDPPolicy):
    def __init__(
        self,
        n_features: int,
        posterior_scale: float = 0.30,
        base_shadow_price: float = 0.22,
        regularization: float = 1.0,
        seed: int = 0,
    ) -> None:
        if posterior_scale <= 0 or base_shadow_price < 0:
            raise ValueError(
                "posterior_scale must be positive and shadow price nonnegative"
            )
        self.posterior_scale = posterior_scale
        self.base_shadow_price = base_shadow_price
        self.regularization = regularization
        super().__init__(n_features, seed)

    def reset(self) -> None:
        super().reset()
        self.model = LinearRidgeModel(self.n_features, self.regularization)

    def scores(self, env: StatefulMicrofinanceCMDPEnv) -> np.ndarray:
        contexts = env.base_context_matrix()
        if len(contexts) == 0:
            return np.array([], dtype=float)
        covariance = (self.posterior_scale**2) * np.linalg.inv(self.model.A)
        theta = self.rng.multivariate_normal(self.model.theta, covariance)
        predicted = contexts @ theta
        costs = env.current_costs()
        median_cost = max(float(np.median(costs)), 1.0)
        normalized_cost = costs / median_cost
        liquidity_pressure = max(
            0.0,
            1.0 - env.available_capital / env.config.initial_capital,
        )
        shadow = self.base_shadow_price * (1.0 + 1.4 * liquidity_pressure)
        return predicted - shadow * normalized_cost

    def decision_context(
        self, env: StatefulMicrofinanceCMDPEnv, action: int
    ) -> np.ndarray:
        return env.current_pool[action].context.copy()

    def update(self, context: np.ndarray, reward: float, cost: float) -> None:
        self.model.update(context, reward)
        self.total_updates += 1


class OraclePortfolioPolicy(CMDPPolicy):
    """Evaluator-only benchmark using hidden repayment probabilities."""

    def __init__(
        self,
        n_features: int,
        state_aware: bool = True,
        base_shadow_price: float = 0.22,
        seed: int = 0,
    ) -> None:
        if base_shadow_price < 0:
            raise ValueError("base_shadow_price cannot be negative")
        self.state_aware = state_aware
        self.base_shadow_price = base_shadow_price
        super().__init__(n_features, seed)

    def scores(self, env: StatefulMicrofinanceCMDPEnv) -> np.ndarray:
        probabilities = env.true_probabilities()
        costs = env.current_costs()
        if len(probabilities) == 0:
            return probabilities
        median_cost = max(float(np.median(costs)), 1.0)
        normalized_cost = costs / median_cost
        if not self.state_aware:
            return probabilities / np.maximum(normalized_cost, 1e-6)
        pressure = max(
            0.0,
            1.0 - env.available_capital / env.config.initial_capital,
        )
        shadow = self.base_shadow_price * (1.0 + 1.4 * pressure)
        return probabilities - shadow * normalized_cost

    def decision_context(
        self, env: StatefulMicrofinanceCMDPEnv, action: int
    ) -> np.ndarray:
        return env.current_pool[action].context.copy()

    def update(self, context: np.ndarray, reward: float, cost: float) -> None:
        self.total_updates += 1


class HardFairCMDPPolicy(CMDPPolicy):
    """Hard quota/parity override around a portfolio-aware base policy."""

    def __init__(
        self,
        base_policy: CMDPPolicy,
        target_shares: Mapping[str, float],
        parity_tolerance: float = 0.05,
        seed: int = 0,
    ) -> None:
        if not 0.0 <= parity_tolerance <= 1.0:
            raise ValueError("parity_tolerance must be in [0, 1]")
        self.base_policy = base_policy
        self.target_shares = dict(target_shares)
        self.parity_tolerance = parity_tolerance
        super().__init__(base_policy.n_features, seed)

    def reset(self) -> None:
        if hasattr(self, "base_policy"):
            self.base_policy.seed = self.seed
            self.base_policy.reset()
        super().reset()

    def scores(self, env: StatefulMicrofinanceCMDPEnv) -> np.ndarray:
        return self.base_policy.scores(env)

    def decision_context(
        self, env: StatefulMicrofinanceCMDPEnv, action: int
    ) -> np.ndarray:
        return self.base_policy.decision_context(env, action)

    def update(self, context: np.ndarray, reward: float, cost: float) -> None:
        self.base_policy.update(context, reward, cost)
        self.total_updates = self.base_policy.total_updates

    def select_action(
        self,
        env: StatefulMicrofinanceCMDPEnv,
        candidates: Sequence[int] | None = None,
    ) -> int | None:
        feasible = sorted(
            env.feasible_actions() if candidates is None else candidates
        )
        if not feasible:
            return None
        groups = env.current_groups()
        next_total = env.total_allocations + 1
        deficits = {
            group: target * next_total
            - env.group_funding_counts.get(group, 0)
            for group, target in self.target_shares.items()
        }
        for group, value in sorted(
            deficits.items(), key=lambda item: item[1], reverse=True
        ):
            if value <= 1e-12:
                continue
            restricted = [
                index for index in feasible if groups[index] == group
            ]
            if restricted:
                return self.base_policy.select_action(env, restricted)

        rates = {
            group: env.group_funding_counts.get(group, 0)
            / max(1, env.group_opportunity_counts.get(group, 0))
            for group in env.group_opportunity_counts
        }
        if rates:
            gap = max(rates.values()) - min(rates.values())
            if gap > self.parity_tolerance:
                minimum = min(rates.values())
                disadvantaged = {
                    group
                    for group, rate in rates.items()
                    if np.isclose(rate, minimum)
                }
                restricted = [
                    index
                    for index in feasible
                    if groups[index] in disadvantaged
                ]
                if restricted:
                    return self.base_policy.select_action(env, restricted)
        return self.base_policy.select_action(env, feasible)


class LagrangianFairCMDPPolicy(CMDPPolicy):
    """Soft fairness wrapper with adaptive quota and parity prices."""

    def __init__(
        self,
        base_policy: CMDPPolicy,
        config: LagrangianConfig,
        seed: int = 0,
    ) -> None:
        self.base_policy = base_policy
        self.config = config
        super().__init__(base_policy.n_features, seed)

    def reset(self) -> None:
        if hasattr(self, "base_policy"):
            self.base_policy.seed = self.seed
            self.base_policy.reset()
        super().reset()
        if hasattr(self, "config"):
            self.lambda_quota = self.config.initial_lambda_quota
            self.lambda_parity = self.config.initial_lambda_parity
        else:
            self.lambda_quota = 0.0
            self.lambda_parity = 0.0
        self.lambda_history: list[tuple[float, float]] = []

    @staticmethod
    def _normalize(scores: np.ndarray, indices: list[int]) -> np.ndarray:
        normalized = np.zeros_like(scores, dtype=float)
        if not indices:
            return normalized
        values = scores[indices]
        low, high = float(np.min(values)), float(np.max(values))
        if np.isclose(low, high):
            normalized[indices] = 0.5
        else:
            normalized[indices] = (values - low) / (high - low)
        return normalized

    def scores(self, env: StatefulMicrofinanceCMDPEnv) -> np.ndarray:
        base_scores = self.base_policy.scores(env)
        feasible = env.feasible_actions()
        scores = self._normalize(base_scores, feasible)
        if env.round_index < self.config.warmup_rounds:
            return scores
        for action in feasible:
            projected = env.projected_fairness(
                action,
                self.config.target_shares,
                self.config.parity_tolerance,
            )
            scores[action] -= (
                self.lambda_quota * projected["quota_violation"]
                + self.lambda_parity * projected["parity_violation"]
            )
        return scores

    def decision_context(
        self, env: StatefulMicrofinanceCMDPEnv, action: int
    ) -> np.ndarray:
        return self.base_policy.decision_context(env, action)

    def update(self, context: np.ndarray, reward: float, cost: float) -> None:
        self.base_policy.update(context, reward, cost)
        self.total_updates = self.base_policy.total_updates

    def on_round_end(self, env: StatefulMicrofinanceCMDPEnv) -> None:
        if env.round_index < self.config.warmup_rounds:
            self.lambda_history.append(
                (self.lambda_quota, self.lambda_parity)
            )
            return
        quota_violation = total_quota_deficit(
            env.group_funding_counts,
            self.config.target_shares,
        )
        parity_gap = demographic_parity_gap(
            env.group_funding_counts,
            env.group_opportunity_counts,
        )
        rate = self.config.dual_learning_rate
        self.lambda_quota = float(
            np.clip(
                self.lambda_quota
                + rate * (quota_violation - self.config.quota_tolerance),
                0.0,
                self.config.max_lambda,
            )
        )
        self.lambda_parity = float(
            np.clip(
                self.lambda_parity
                + rate * (parity_gap - self.config.parity_tolerance),
                0.0,
                self.config.max_lambda,
            )
        )
        self.lambda_history.append((self.lambda_quota, self.lambda_parity))
