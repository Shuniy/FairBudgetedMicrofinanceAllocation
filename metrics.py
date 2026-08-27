from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


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
    """Computes the funding rates for each group as the fraction of funded counts over opportunity counts.
    If a group has no opportunities, its funding rate is considered to be 0.0."""
    groups = set(funded) | set(opportunities)
    return {
        group: (
            funded.get(group, 0) / opportunities[group]
            if opportunities.get(group, 0) > 0
            else 0.0
        )
        for group in groups
    }


def demographic_parity_gap(
    funded: Mapping[str, int],
    opportunities: Mapping[str, int] | None = None,
) -> float:
    """Important metric to check. Computes the demographic parity gap as the difference between the maximum and minimum allocation shares or funding rates across groups.
    If opportunities are provided, the funding rates are used; otherwise, allocation shares are used."""
    rates = (
        allocation_shares(funded)
        if opportunities is None
        else group_funding_rates(funded, opportunities)
    )
    return float(max(rates.values()) - min(rates.values())) if rates else 0.0


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
    """Important factor to check. Computes the quota satisfaction rate as the fraction of groups whose allocation shares meet or exceed their target shares.
    If there are no target shares, returns 1.0."""
    if not target_shares:
        return 1.0
    deficits = quota_deficits(funded, target_shares)
    return sum(value <= 1e-12 for value in deficits.values()) / len(deficits)


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
    """Return the amount by which the opportunity-based parity gap exceeds a limit."""

    return max(0.0, demographic_parity_gap(funded, opportunities) - tolerance)


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
