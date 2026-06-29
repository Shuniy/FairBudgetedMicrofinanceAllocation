from typing import Mapping, Sequence

import numpy as np

from Algorithms.bandit import BanditPolicy
from Environments.synthetic_environment_bandits import SyntheticEnvironment


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

    def scores(self, env: SyntheticEnvironment) -> np.ndarray:
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
