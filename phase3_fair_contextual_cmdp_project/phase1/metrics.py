from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def allocation_shares(
    group_counts: Mapping[str, int],
) -> dict[str, float]:
    total = sum(group_counts.values())
    if total == 0:
        return {group: 0.0 for group in group_counts}
    return {group: count / total for group, count in group_counts.items()}


def demographic_parity_gap(
    group_counts: Mapping[str, int],
    opportunities: Mapping[str, int] | None = None,
) -> float:
    """Maximum difference between observed group funding rates.

    In the Phase-1 group-arm environment, every group is considered available
    at every round. Therefore, this reduces to the gap between group allocation
    shares. In later phases, pass applicant opportunity counts explicitly.
    """

    if not group_counts:
        return 0.0

    if opportunities is None:
        rates = list(allocation_shares(group_counts).values())
    else:
        rates = [
            group_counts.get(group, 0) / max(1, opportunities.get(group, 0))
            for group in group_counts
        ]
    return float(max(rates) - min(rates)) if rates else 0.0


def quota_deficits(
    group_counts: Mapping[str, int],
    target_shares: Mapping[str, float],
) -> dict[str, float]:
    shares = allocation_shares(group_counts)
    return {
        group: max(0.0, target - shares.get(group, 0.0))
        for group, target in target_shares.items()
    }


def quota_satisfaction_rate(
    group_counts: Mapping[str, int],
    target_shares: Mapping[str, float],
) -> float:
    if not target_shares:
        return 1.0
    deficits = quota_deficits(group_counts, target_shares)
    satisfied = sum(deficit <= 1e-12 for deficit in deficits.values())
    return satisfied / len(deficits)


def budget_utilization(initial_budget: float, remaining_budget: float) -> float:
    return (initial_budget - remaining_budget) / initial_budget


def reward_per_unit_cost(rewards: Sequence[float], costs: Sequence[float]) -> float:
    total_cost = float(np.sum(costs))
    return float(np.sum(rewards) / total_cost) if total_cost > 0 else 0.0
