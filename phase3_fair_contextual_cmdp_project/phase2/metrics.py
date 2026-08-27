from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def allocation_shares(group_counts: Mapping[str, int]) -> dict[str, float]:
    total = sum(group_counts.values())
    return {
        group: (count / total if total else 0.0)
        for group, count in group_counts.items()
    }


def group_funding_rates(
    funded: Mapping[str, int],
    opportunities: Mapping[str, int],
) -> dict[str, float]:
    groups = set(funded) | set(opportunities)
    return {
        group: funded.get(group, 0) / max(1, opportunities.get(group, 0))
        for group in groups
    }


def demographic_parity_gap(
    funded: Mapping[str, int],
    opportunities: Mapping[str, int],
) -> float:
    rates = list(group_funding_rates(funded, opportunities).values())
    return float(max(rates) - min(rates)) if rates else 0.0


def quota_deficits(
    funded: Mapping[str, int],
    target_shares: Mapping[str, float],
) -> dict[str, float]:
    shares = allocation_shares(funded)
    return {
        group: max(0.0, target - shares.get(group, 0.0))
        for group, target in target_shares.items()
    }


def quota_satisfaction_rate(
    funded: Mapping[str, int],
    target_shares: Mapping[str, float],
) -> float:
    if not target_shares:
        return 1.0
    deficits = quota_deficits(funded, target_shares)
    return sum(value <= 1e-12 for value in deficits.values()) / len(deficits)


def budget_utilization(initial_budget: float, remaining_budget: float) -> float:
    return (initial_budget - remaining_budget) / initial_budget


def reward_per_unit_cost(rewards: Sequence[float], costs: Sequence[float]) -> float:
    total_cost = float(np.sum(costs))
    return float(np.sum(rewards) / total_cost) if total_cost > 0 else 0.0


def repayment_rate(rewards: Sequence[float]) -> float:
    return float(np.mean(rewards)) if len(rewards) else 0.0
