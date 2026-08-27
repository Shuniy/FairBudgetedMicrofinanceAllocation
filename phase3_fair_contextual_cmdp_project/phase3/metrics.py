from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from phase2.metrics import (
    allocation_shares,
    demographic_parity_gap,
    group_funding_rates,
    quota_deficits,
    quota_satisfaction_rate,
)


def total_quota_deficit(
    funded: Mapping[str, int],
    target_shares: Mapping[str, float],
) -> float:
    return float(sum(quota_deficits(funded, target_shares).values()))


def parity_violation(
    funded: Mapping[str, int],
    opportunities: Mapping[str, int],
    tolerance: float,
) -> float:
    return max(0.0, demographic_parity_gap(funded, opportunities) - tolerance)


def repayment_rate(rewards: Sequence[float]) -> float:
    return float(np.mean(rewards)) if len(rewards) else 0.0


def reward_per_disbursed_capital(
    rewards: Sequence[float],
    principals: Sequence[float],
) -> float:
    total = float(np.sum(principals))
    return float(np.sum(rewards) / total) if total > 0 else 0.0


def capital_turnover(cumulative_disbursed: float, initial_capital: float) -> float:
    return cumulative_disbursed / initial_capital


def liquidity_utilization(available_capital: float, capital_at_risk: float) -> float:
    surviving = available_capital + capital_at_risk
    return capital_at_risk / surviving if surviving > 0 else 0.0


def capital_survival_ratio(available_capital: float, capital_at_risk: float, initial: float) -> float:
    return (available_capital + capital_at_risk) / initial


__all__ = [
    "allocation_shares",
    "group_funding_rates",
    "demographic_parity_gap",
    "quota_deficits",
    "quota_satisfaction_rate",
    "total_quota_deficit",
    "parity_violation",
    "repayment_rate",
    "reward_per_disbursed_capital",
    "capital_turnover",
    "liquidity_utilization",
    "capital_survival_ratio",
]
