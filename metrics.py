from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import nan

import numpy as np

QUOTA_NUMERICAL_TOLERANCE = 1e-12


def allocation_shares(group_counts: Mapping[str, int]) -> dict[str, float]:
    """Computes the allocation shares for each group as the fraction of counts over the total counts.
    If there are no counts, all shares are considered to be 0.0."""
    total = sum(group_counts.values())
    return {
        group: (count / total if total else 0.0)
        for group, count in group_counts.items()
    }


def group_funding_rates(
    funded: Mapping[str, int],
    opportunities: Mapping[str, int],
) -> dict[str, float]:
    """Compute funded / opportunities for groups with positive opportunities."""
    return {
        group: funded.get(group, 0) / count
        for group, count in opportunities.items()
        if count > 0
    }


def allocation_share_gap(funded: Mapping[str, int]) -> float:
    """Compute the spread of allocation shares across all supplied groups."""
    shares = allocation_shares(funded)
    return max(shares.values()) - min(shares.values()) if shares else 0.0


def demographic_parity_gap(
    funded: Mapping[str, int],
    opportunities: Mapping[str, int] | None = None,
) -> float:
    """Compute the funding-rate spread, undefined with fewer than two rates."""
    if opportunities is None:
        return allocation_share_gap(funded)  # Legacy allocation-share call.
    rates = group_funding_rates(funded, opportunities)
    return max(rates.values()) - min(rates.values()) if len(rates) >= 2 else nan


def quota_deficits(
    funded: Mapping[str, int],
    target_shares: Mapping[str, float],
) -> dict[str, float]:
    """Computes the quota deficits for each group as the difference between the target share and the actual allocation share.
    If a group is not present in the funded counts, its allocation share is considered to be 0.0. If a group is not present in the target shares, it is ignored."""
    shares = allocation_shares(funded)
    return {
        group: max(0.0, target - shares.get(group, 0.0))
        for group, target in target_shares.items()
    }


def quota_satisfaction_rate(
    funded: Mapping[str, int],
    target_shares: Mapping[str, float],
) -> float:
    """Return the fraction of positive quotas met within numerical tolerance."""
    deficits = [
        deficit
        for group, deficit in quota_deficits(funded, target_shares).items()
        if target_shares[group] > 0
    ]
    if not deficits:
        return 1.0
    return sum(value <= QUOTA_NUMERICAL_TOLERANCE for value in deficits) / len(deficits)


def budget_utilization(initial_budget: float, remaining_budget: float) -> float:
    """Computes the budget utilization as the fraction of the initial budget that has been spent.
    If the initial budget is zero, returns 0.0."""
    return (initial_budget - remaining_budget) / initial_budget


def reward_per_unit_cost(rewards: Sequence[float], costs: Sequence[float]) -> float:
    """Important but not that much. Computes the reward per unit cost as the total rewards divided by the total costs.
    If there are no costs, returns 0.0."""
    total_cost = float(np.sum(costs))
    return float(np.sum(rewards) / total_cost) if total_cost > 0 else 0.0


def repayment_rate(rewards: Sequence[float]) -> float:
    """
    Computes the repayment rate as the mean of the rewards.
    If there are no rewards, returns 0.0."""
    return float(np.mean(rewards)) if len(rewards) else 0.0


def total_quota_deficit(
    funded: Mapping[str, int],
    target_shares: Mapping[str, float],
) -> float:
    """Return the total shortfall across all configured allocation quotas."""

    return float(sum(quota_deficits(funded, target_shares).values()))


def parity_violation(
    funded: Mapping[str, int],
    opportunities: Mapping[str, int],
    tolerance: float,
) -> float:
    """Return positive parity excess, with no penalty for an undefined gap."""

    gap = demographic_parity_gap(funded, opportunities)
    return gap - tolerance if gap > tolerance else 0.0


def reward_per_disbursed_capital(
    rewards: Sequence[float],
    principals: Sequence[float],
) -> float:
    """Phase 3 name for the shared reward-per-cost calculation."""

    return reward_per_unit_cost(rewards, principals)


def capital_turnover(cumulative_disbursed: float, initial_capital: float) -> float:
    """Measure how many times the initial capital stock has been lent."""

    return cumulative_disbursed / initial_capital if initial_capital > 0 else 0.0


def liquidity_utilization(available_capital: float, capital_at_risk: float) -> float:
    """Return the share of surviving capital currently committed to loans."""

    surviving = available_capital + capital_at_risk
    return capital_at_risk / surviving if surviving > 0 else 0.0


def capital_survival_ratio(
    available_capital: float,
    capital_at_risk: float,
    initial_capital: float,
) -> float:
    """Return current liquid plus at-risk capital relative to initial capital."""

    return (
        (available_capital + capital_at_risk) / initial_capital
        if initial_capital > 0
        else 0.0
    )
