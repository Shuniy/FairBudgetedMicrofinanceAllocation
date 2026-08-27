from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Mapping, Sequence

import numpy as np

from .environment import SyntheticMicrofinanceEnv


class BanditPolicy(ABC):
    """Base class shared by all Phase-1 policies."""

    def __init__(self, n_arms: int, seed: int = 0) -> None:
        if n_arms <= 0:
            raise ValueError("n_arms must be positive")
        self.n_arms = n_arms
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.counts = np.zeros(self.n_arms, dtype=int)
        self.reward_sums = np.zeros(self.n_arms, dtype=float)
        self.cost_sums = np.zeros(self.n_arms, dtype=float)
        self.total_updates = 0

    @property
    def mean_rewards(self) -> np.ndarray:
        return np.divide(
            self.reward_sums,
            self.counts,
            out=np.zeros_like(self.reward_sums),
            where=self.counts > 0,
        )

    @property
    def mean_costs(self) -> np.ndarray:
        return np.divide(
            self.cost_sums,
            self.counts,
            out=np.full_like(self.cost_sums, np.inf),
            where=self.counts > 0,
        )

    def update(self, action: int, reward: float, cost: float) -> None:
        self.counts[action] += 1
        self.reward_sums[action] += reward
        self.cost_sums[action] += cost
        self.total_updates += 1

    @abstractmethod
    def scores(self, env: SyntheticMicrofinanceEnv) -> np.ndarray:
        """Return one action score per arm."""

    def select_action(
        self,
        env: SyntheticMicrofinanceEnv,
        candidates: Sequence[int] | None = None,
    ) -> int:
        feasible = list(env.feasible_actions() if candidates is None else candidates)
        feasible = [action for action in feasible if action in env.feasible_actions()]
        if not feasible:
            raise RuntimeError("No feasible action available")

        action_scores = self.scores(env)
        feasible_scores = np.asarray([action_scores[a] for a in feasible], dtype=float)
        best_score = np.nanmax(feasible_scores)
        best = [
            action
            for action, score in zip(feasible, feasible_scores, strict=True)
            if np.isclose(score, best_score) or (np.isinf(score) and np.isinf(best_score))
        ]
        return int(self.rng.choice(best))


class RandomPolicy(BanditPolicy):
    def scores(self, env: SyntheticMicrofinanceEnv) -> np.ndarray:
        return self.rng.random(self.n_arms)


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

    def scores(self, env: SyntheticMicrofinanceEnv) -> np.ndarray:
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

    def scores(self, env: SyntheticMicrofinanceEnv) -> np.ndarray:
        self.epsilon = max(
            self.minimum_epsilon,
            self.initial_epsilon / (1.0 + self.decay_rate * self.total_updates),
        )
        return super().scores(env)


class UCB1Policy(BanditPolicy):
    def __init__(self, n_arms: int, exploration_coefficient: float = 2.0, seed: int = 0) -> None:
        if exploration_coefficient < 0:
            raise ValueError("exploration_coefficient cannot be negative")
        self.exploration_coefficient = exploration_coefficient
        super().__init__(n_arms=n_arms, seed=seed)

    def scores(self, env: SyntheticMicrofinanceEnv) -> np.ndarray:
        scores = np.full(self.n_arms, np.inf, dtype=float)
        sampled = self.counts > 0
        if np.any(sampled):
            bonus = np.sqrt(
                self.exploration_coefficient
                * np.log(max(2, self.total_updates + 1))
                / self.counts[sampled]
            )
            scores[sampled] = self.mean_rewards[sampled] + bonus
        return scores


class ThompsonSamplingPolicy(BanditPolicy):
    def reset(self) -> None:
        super().reset()
        self.alpha = np.ones(self.n_arms, dtype=float)
        self.beta = np.ones(self.n_arms, dtype=float)

    def update(self, action: int, reward: float, cost: float) -> None:
        super().update(action, reward, cost)
        self.alpha[action] += reward
        self.beta[action] += 1.0 - reward

    def scores(self, env: SyntheticMicrofinanceEnv) -> np.ndarray:
        return self.rng.beta(self.alpha, self.beta)


class RewardPerCostPolicy(BanditPolicy):
    """Greedy estimated repayment per unit of capital."""

    def scores(self, env: SyntheticMicrofinanceEnv) -> np.ndarray:
        scores = np.full(self.n_arms, np.inf, dtype=float)
        sampled = self.counts > 0
        if np.any(sampled):
            # Current requested loan cost is known before funding.
            scores[sampled] = self.mean_rewards[sampled] / env.current_costs[sampled]
        return scores


class BudgetedUCBPolicy(UCB1Policy):
    """UCB reward estimate divided by the current known loan cost."""

    def scores(self, env: SyntheticMicrofinanceEnv) -> np.ndarray:
        reward_ucb = super().scores(env)
        return reward_ucb / env.current_costs


class EpsilonFirstPolicy(BanditPolicy):
    """Spend a fixed budget fraction exploring, then exploit reward/cost."""

    def __init__(
        self,
        n_arms: int,
        exploration_budget_fraction: float = 0.2,
        seed: int = 0,
    ) -> None:
        if not 0.0 < exploration_budget_fraction < 1.0:
            raise ValueError("exploration_budget_fraction must be between 0 and 1")
        self.exploration_budget_fraction = exploration_budget_fraction
        super().__init__(n_arms=n_arms, seed=seed)

    def scores(self, env: SyntheticMicrofinanceEnv) -> np.ndarray:
        threshold = self.exploration_budget_fraction * env.config.total_budget
        if env.spent_budget < threshold:
            # Balanced exploration: prefer the least sampled arm.
            return -self.counts.astype(float)

        scores = np.full(self.n_arms, -np.inf, dtype=float)
        sampled = self.counts > 0
        scores[sampled] = self.mean_rewards[sampled] / env.current_costs[sampled]
        if not np.any(sampled):
            return self.rng.random(self.n_arms)
        return scores


class FractionalKUBEPolicy(UCB1Policy):
    """Fractional-knapsack approximation: optimistic reward/cost ratio."""

    def scores(self, env: SyntheticMicrofinanceEnv) -> np.ndarray:
        optimistic_rewards = super().scores(env)
        return optimistic_rewards / env.current_costs


class KUBEPolicy(UCB1Policy):
    """Small unbounded-knapsack KUBE-style policy.

    This implementation is intended for integer or near-integer deterministic
    costs. It solves an unbounded knapsack over the remaining budget using UCB
    reward estimates, then samples an arm in proportion to the recovered item
    counts. For stochastic costs, use FractionalKUBEPolicy or BudgetedUCBPolicy.
    """

    def __init__(
        self,
        n_arms: int,
        exploration_coefficient: float = 2.0,
        cost_scale: int = 1,
        seed: int = 0,
    ) -> None:
        if cost_scale <= 0:
            raise ValueError("cost_scale must be positive")
        self.cost_scale = cost_scale
        super().__init__(n_arms, exploration_coefficient, seed)

    def scores(self, env: SyntheticMicrofinanceEnv) -> np.ndarray:
        # Used only by wrappers. The direct KUBE action selection is below.
        return super().scores(env) / env.current_costs

    def select_action(
        self,
        env: SyntheticMicrofinanceEnv,
        candidates: Sequence[int] | None = None,
    ) -> int:
        feasible = list(env.feasible_actions() if candidates is None else candidates)
        feasible = [action for action in feasible if action in env.feasible_actions()]
        if not feasible:
            raise RuntimeError("No feasible action available")

        untried = [action for action in feasible if self.counts[action] == 0]
        if untried:
            return int(self.rng.choice(untried))

        budget_units = int(np.floor(env.remaining_budget * self.cost_scale + 1e-9))
        costs = {
            action: max(1, int(round(env.current_costs[action] * self.cost_scale)))
            for action in feasible
        }
        values = super().scores(env)

        dp = np.zeros(budget_units + 1, dtype=float)
        choice = np.full(budget_units + 1, -1, dtype=int)
        for capacity in range(1, budget_units + 1):
            best_value = dp[capacity]
            best_action = -1
            for action in feasible:
                item_cost = costs[action]
                if item_cost <= capacity:
                    candidate_value = dp[capacity - item_cost] + values[action]
                    if candidate_value > best_value + 1e-12:
                        best_value = candidate_value
                        best_action = action
            dp[capacity] = best_value
            choice[capacity] = best_action

        selected_counts: Counter[int] = Counter()
        capacity = budget_units
        while capacity > 0 and choice[capacity] != -1:
            action = int(choice[capacity])
            selected_counts[action] += 1
            capacity -= costs[action]

        if not selected_counts:
            return super().select_action(env, feasible)

        actions = np.asarray(list(selected_counts.keys()), dtype=int)
        probabilities = np.asarray(list(selected_counts.values()), dtype=float)
        probabilities /= probabilities.sum()
        return int(self.rng.choice(actions, p=probabilities))


class OracleRewardPerCostPolicy(BanditPolicy):
    """Evaluation-only policy that knows the true repayment probabilities."""

    def scores(self, env: SyntheticMicrofinanceEnv) -> np.ndarray:
        probabilities = np.asarray(
            [arm.repayment_probability for arm in env.arms], dtype=float
        )
        return probabilities / env.current_costs


class FairnessConstrainedPolicy(BanditPolicy):
    """Hard fairness wrapper with priority: quota, then parity, then utility.

    The wrapped base policy retains its own learning state. This wrapper only
    restricts which feasible actions the base policy may choose.
    """

    def __init__(
        self,
        base_policy: BanditPolicy,
        arm_groups: Sequence[str],
        target_shares: Mapping[str, float],
        parity_tolerance: float = 1.0,
        seed: int = 0,
    ) -> None:
        if len(arm_groups) != base_policy.n_arms:
            raise ValueError("arm_groups length must match base policy arms")
        if sum(target_shares.values()) > 1.0 + 1e-12:
            raise ValueError("target shares cannot sum to more than 1")
        if not 0.0 <= parity_tolerance <= 1.0:
            raise ValueError("parity_tolerance must be in [0, 1]")
        self.base_policy = base_policy
        self.arm_groups = tuple(arm_groups)
        self.target_shares = dict(target_shares)
        self.parity_tolerance = parity_tolerance
        super().__init__(n_arms=base_policy.n_arms, seed=seed)

    def reset(self) -> None:
        # Guard for BanditPolicy.__init__, which calls reset.
        if hasattr(self, "base_policy"):
            self.base_policy.reset()
        super().reset()

    @property
    def counts(self) -> np.ndarray:  # type: ignore[override]
        return self.base_policy.counts

    @counts.setter
    def counts(self, value: np.ndarray) -> None:
        self._wrapper_counts = value

    @property
    def reward_sums(self) -> np.ndarray:  # type: ignore[override]
        return self.base_policy.reward_sums

    @reward_sums.setter
    def reward_sums(self, value: np.ndarray) -> None:
        self._wrapper_reward_sums = value

    @property
    def cost_sums(self) -> np.ndarray:  # type: ignore[override]
        return self.base_policy.cost_sums

    @cost_sums.setter
    def cost_sums(self, value: np.ndarray) -> None:
        self._wrapper_cost_sums = value

    @property
    def total_updates(self) -> int:  # type: ignore[override]
        return self.base_policy.total_updates

    @total_updates.setter
    def total_updates(self, value: int) -> None:
        self._wrapper_total_updates = value

    def update(self, action: int, reward: float, cost: float) -> None:
        self.base_policy.update(action, reward, cost)

    def scores(self, env: SyntheticMicrofinanceEnv) -> np.ndarray:
        return self.base_policy.scores(env)

    def _actions_for_groups(
        self,
        feasible: Sequence[int],
        groups: set[str],
    ) -> list[int]:
        return [action for action in feasible if self.arm_groups[action] in groups]

    def select_action(
        self,
        env: SyntheticMicrofinanceEnv,
        candidates: Sequence[int] | None = None,
    ) -> int:
        feasible = list(env.feasible_actions() if candidates is None else candidates)
        feasible = [action for action in feasible if action in env.feasible_actions()]
        if not feasible:
            raise RuntimeError("No feasible action available")

        t_next = env.total_allocations + 1
        quota_deficits = {
            group: target * t_next - env.group_allocation_counts.get(group, 0)
            for group, target in self.target_shares.items()
        }
        positive = {group: deficit for group, deficit in quota_deficits.items() if deficit > 1e-12}
        if positive:
            maximum = max(positive.values())
            most_underrepresented = {
                group for group, deficit in positive.items() if np.isclose(deficit, maximum)
            }
            restricted = self._actions_for_groups(feasible, most_underrepresented)
            if restricted:
                return self.base_policy.select_action(env, restricted)

        counts = env.group_allocation_counts
        total = max(1, env.total_allocations)
        shares = {group: count / total for group, count in counts.items()}
        if shares and max(shares.values()) - min(shares.values()) > self.parity_tolerance:
            minimum_share = min(shares.values())
            least_funded = {
                group for group, share in shares.items() if np.isclose(share, minimum_share)
            }
            restricted = self._actions_for_groups(feasible, least_funded)
            if restricted:
                return self.base_policy.select_action(env, restricted)

        return self.base_policy.select_action(env, feasible)


class FairnessPenaltyPolicy(BanditPolicy):
    """Soft fairness policy: utility score plus a quota-deficit bonus."""

    def __init__(
        self,
        base_policy: BanditPolicy,
        arm_groups: Sequence[str],
        target_shares: Mapping[str, float],
        penalty_lambda: float,
        seed: int = 0,
    ) -> None:
        if penalty_lambda < 0:
            raise ValueError("penalty_lambda cannot be negative")
        self.base_policy = base_policy
        self.arm_groups = tuple(arm_groups)
        self.target_shares = dict(target_shares)
        self.penalty_lambda = penalty_lambda
        super().__init__(n_arms=base_policy.n_arms, seed=seed)

    def reset(self) -> None:
        if hasattr(self, "base_policy"):
            self.base_policy.reset()
        super().reset()

    @property
    def counts(self) -> np.ndarray:  # type: ignore[override]
        return self.base_policy.counts

    @counts.setter
    def counts(self, value: np.ndarray) -> None:
        self._wrapper_counts = value

    @property
    def reward_sums(self) -> np.ndarray:  # type: ignore[override]
        return self.base_policy.reward_sums

    @reward_sums.setter
    def reward_sums(self, value: np.ndarray) -> None:
        self._wrapper_reward_sums = value

    @property
    def cost_sums(self) -> np.ndarray:  # type: ignore[override]
        return self.base_policy.cost_sums

    @cost_sums.setter
    def cost_sums(self, value: np.ndarray) -> None:
        self._wrapper_cost_sums = value

    @property
    def total_updates(self) -> int:  # type: ignore[override]
        return self.base_policy.total_updates

    @total_updates.setter
    def total_updates(self, value: int) -> None:
        self._wrapper_total_updates = value

    def update(self, action: int, reward: float, cost: float) -> None:
        self.base_policy.update(action, reward, cost)

    def scores(self, env: SyntheticMicrofinanceEnv) -> np.ndarray:
        utility = self.base_policy.scores(env).copy()
        total = max(1, env.total_allocations)
        shares = {
            group: count / total for group, count in env.group_allocation_counts.items()
        }
        for action, group in enumerate(self.arm_groups):
            target = self.target_shares.get(group, 0.0)
            deficit = max(0.0, target - shares.get(group, 0.0))
            if np.isfinite(utility[action]):
                utility[action] += self.penalty_lambda * deficit
        return utility
