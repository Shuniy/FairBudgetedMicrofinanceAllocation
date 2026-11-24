from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from Algorithms.bandit import BasePolicy
from Environments.base_environment import BanditEnvironment, BanditMode
from metrics import group_funding_rates, quota_deficits
from models.events import FeedbackEvent


class _FairPolicy(BasePolicy):
    """Base class for fairness-aware bandit policies."""
    def __init__(
        self,
        base_policy: BasePolicy,
        target_shares: Mapping[str, float],
        seed: int = 0,
    ) -> None:
        self.base_policy = base_policy
        self.target_shares = dict(target_shares)
        if any(not 0.0 <= value <= 1.0 for value in self.target_shares.values()):
            raise ValueError("target shares must be in [0, 1]")
        if sum(self.target_shares.values()) > 1.0 + 1e-12:
            raise ValueError("target shares cannot sum to more than one")
        super().__init__(seed)

    def reset(self) -> None:
        if hasattr(self, "base_policy"):
            self.base_policy.seed = self.seed
            self.base_policy.reset()
        super().reset()

    def scores(self, env: BanditEnvironment) -> np.ndarray:
        return self.base_policy.scores(env)

    def update(self, feedback: FeedbackEvent) -> None:
        self.base_policy.update(feedback)
        self.total_updates = self.base_policy.total_updates


class HardFairPolicy(_FairPolicy):
    """Quota first, funding-rate parity second, base utility third."""

    def __init__(
        self,
        base_policy: BasePolicy,
        target_shares: Mapping[str, float],
        parity_tolerance: float = 1.0,
        seed: int = 0,
    ) -> None:
        """Initialize the HardFairPolicy with a base policy, target shares, and parity tolerance."""
        if not 0.0 <= parity_tolerance <= 1.0:
            raise ValueError("parity_tolerance must be in [0, 1]")
        self.parity_tolerance = parity_tolerance
        super().__init__(base_policy, target_shares, seed)

    def select_action(
        self,
        env: BanditEnvironment,
        candidates: Sequence[int] | None = None,
    ) -> int:
        """Select an action based on quota deficits, funding-rate parity, and base utility."""
        feasible_set = set(env.feasible_actions())
        feasible = sorted(feasible_set if candidates is None else candidates)
        feasible = [index for index in feasible if index in feasible_set]
        groups = env.current_groups()

        next_total = env.total_allocations + 1
        deficits = {
            group: target * next_total - env.group_funding_counts.get(group, 0)
            for group, target in self.target_shares.items()
        }
        for group, deficit in sorted(
            deficits.items(), key=lambda item: item[1], reverse=True
        ):
            if deficit <= 1e-12:
                continue
            restricted = [index for index in feasible if groups[index] == group]
            if restricted:
                return self.base_policy.select_action(env, restricted)

        rates = group_funding_rates(
            env.group_funding_counts, env.group_opportunity_counts
        )
        if env.mode == BanditMode.ARM:
            # Preserve the arm intervention's affordable group-round rates.
            rates = {group: rates.get(group, 0.0) for group in env.groups}
        if (
            len(rates) >= 2
            and max(rates.values()) - min(rates.values()) > self.parity_tolerance
        ):
            minimum = min(rates.values())
            disadvantaged = {
                group for group, rate in rates.items() if np.isclose(rate, minimum)
            }
            restricted = [
                index for index in feasible if groups[index] in disadvantaged
            ]
            if restricted:
                return self.base_policy.select_action(env, restricted)

        return self.base_policy.select_action(env, feasible)


class SoftFairPolicy(_FairPolicy):
    """Add a current quota-shortfall bonus to finite base scores."""
    def __init__(
        self,
        base_policy: BasePolicy,
        target_shares: Mapping[str, float],
        penalty_lambda: float,
        seed: int = 0,
    ) -> None:
        if penalty_lambda < 0:
            raise ValueError("penalty_lambda cannot be negative")
        self.penalty_lambda = penalty_lambda
        super().__init__(base_policy, target_shares, seed)

    def select_action(
        self,
        env: BanditEnvironment,
        candidates: Sequence[int] | None = None,
    ) -> int:
        if self.penalty_lambda == 0:
            return self.base_policy.select_action(env, candidates)
        return super().select_action(env, candidates)

    def scores(self, env: BanditEnvironment) -> np.ndarray:
        """Compute scores with bonuses for quota deficits."""
        scores = self.base_policy.scores(env).copy()
        deficits = quota_deficits(env.group_funding_counts, self.target_shares)
        for index, group in enumerate(env.current_groups()):
            if np.isfinite(scores[index]):
                scores[index] += self.penalty_lambda * deficits.get(group, 0.0)
        return scores
