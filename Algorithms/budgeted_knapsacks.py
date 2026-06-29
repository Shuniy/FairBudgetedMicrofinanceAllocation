import numpy as np


class BudgetedUCB:
    def __init__(self, n_arms, budget, c=2.0):
        self.n_arms = n_arms
        self.initial_budget = budget
        self.remaining_budget = budget
        self.c = c

        self.counts = np.zeros(n_arms, dtype=int)
        self.reward_estimates = np.zeros(n_arms)
        self.cost_estimates = np.ones(n_arms)  # avoid division by zero

        self.t = 0

    def select_arm(self):
        self.t += 1

        # Pull each arm at least once
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm

        scores = np.zeros(self.n_arms)

        for arm in range(self.n_arms):
            reward_ucb = self.reward_estimates[arm] + np.sqrt(
                self.c * np.log(self.t) / self.counts[arm]
            )

            expected_cost = max(self.cost_estimates[arm], 1e-8)

            # Budgeted score: reward per unit cost
            scores[arm] = reward_ucb / expected_cost

        return int(np.argmax(scores))

    def update(self, arm, reward, cost):
        self.counts[arm] += 1
        n = self.counts[arm]

        self.reward_estimates[arm] += (reward - self.reward_estimates[arm]) / n

        self.cost_estimates[arm] += (cost - self.cost_estimates[arm]) / n

        self.remaining_budget -= cost
