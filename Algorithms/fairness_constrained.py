from typing import Mapping, Sequence

import numpy as np

from Algorithms.bandit import BanditPolicy
from Environments.synthetic_environment_bandits import SyntheticEnvironment


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

    def scores(self, env: SyntheticEnvironment) -> np.ndarray:
        return self.base_policy.scores(env)

    def _actions_for_groups(
        self,
        feasible: Sequence[int],
        groups: set[str],
    ) -> list[int]:
        return [action for action in feasible if self.arm_groups[action] in groups]

    def select_action(
        self,
        env: SyntheticEnvironment,
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
        positive = {
            group: deficit
            for group, deficit in quota_deficits.items()
            if deficit > 1e-12
        }
        if positive:
            maximum = max(positive.values())
            most_underrepresented = {
                group
                for group, deficit in positive.items()
                if np.isclose(deficit, maximum)
            }
            restricted = self._actions_for_groups(feasible, most_underrepresented)
            if restricted:
                return self.base_policy.select_action(env, restricted)

        counts = env.group_allocation_counts
        total = max(1, env.total_allocations)
        shares = {group: count / total for group, count in counts.items()}
        if (
            shares
            and max(shares.values()) - min(shares.values()) > self.parity_tolerance
        ):
            minimum_share = min(shares.values())
            least_funded = {
                group
                for group, share in shares.items()
                if np.isclose(share, minimum_share)
            }
            restricted = self._actions_for_groups(feasible, least_funded)
            if restricted:
                return self.base_policy.select_action(env, restricted)

        return self.base_policy.select_action(env, feasible)
