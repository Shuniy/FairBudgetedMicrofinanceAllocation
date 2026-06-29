from collections import Counter
from typing import Sequence

import numpy as np

from Algorithms.ucb import UCBPolicy
from Environments.synthetic_environment_bandits import SyntheticEnvironment


class KUBEPolicy(UCBPolicy):
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

    def scores(self, env: SyntheticEnvironment) -> np.ndarray:
        # Used only by wrappers. The direct KUBE action selection is below.
        return super().scores(env) / env.current_costs

    def select_action(
        self,
        env: SyntheticEnvironment,
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
