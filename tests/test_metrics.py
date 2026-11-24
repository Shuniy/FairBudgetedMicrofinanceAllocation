from __future__ import annotations

from math import isnan

import pytest

from metrics import (
    allocation_shares,
    allocation_share_gap,
    demographic_parity_gap,
    group_funding_rates,
    quota_deficits,
    quota_satisfaction_rate,
    total_quota_deficit,
)


def test_metrics_handle_zero_and_missing_opportunities() -> None:
    rates = group_funding_rates({"A": 1, "B": 0}, {"A": 2, "B": 0})
    assert rates == {"A": 0.5}
    assert isnan(demographic_parity_gap({"A": 1, "B": 0}, {"A": 2, "B": 0}))
    assert isnan(demographic_parity_gap({}, {}))


def test_arm_allocation_share_gap_includes_unfunded_groups() -> None:
    counts = {"A": 50, "B": 30, "C": 20}
    assert allocation_shares(counts) == {"A": 0.50, "B": 0.30, "C": 0.20}
    assert allocation_share_gap(counts) == pytest.approx(0.30)
    assert allocation_share_gap({**counts, "D": 0}) == 0.50
    assert allocation_share_gap({"A": 0, "B": 0}) == 0


def test_population_and_feasible_parity_differ_with_same_funded_counts() -> None:
    funded = {"A": 20, "B": 10}
    population = {"A": 100, "B": 100}
    feasible = {"A": 100, "B": 50}
    assert group_funding_rates(funded, population) == {"A": 0.20, "B": 0.10}
    assert group_funding_rates(funded, feasible) == {"A": 0.20, "B": 0.20}
    assert demographic_parity_gap(funded, population) == pytest.approx(0.10)
    assert demographic_parity_gap(funded, feasible) == 0.0


def test_quota_metrics_use_allocation_shares() -> None:
    counts = {"A": 3, "B": 1}
    assert allocation_shares(counts) == {"A": 0.75, "B": 0.25}
    assert quota_deficits(counts, {"A": 0.5, "B": 0.5}) == {
        "A": 0.0,
        "B": 0.25,
    }
    assert quota_satisfaction_rate(counts, {"A": 0.5, "B": 0.5}) == pytest.approx(0.5)


def test_quota_deficits_and_positive_target_satisfaction() -> None:
    funded = {"A": 50, "B": 30, "C": 20}
    quotas = {"A": 0.20, "B": 0.30, "C": 0.30, "D": 0.0}
    assert quota_deficits(funded, quotas) == pytest.approx(
        {"A": 0.0, "B": 0.0, "C": 0.10, "D": 0.0}
    )
    assert total_quota_deficit(funded, quotas) == pytest.approx(0.10)
    assert quota_satisfaction_rate(funded, quotas) == pytest.approx(2 / 3)
    assert quota_deficits({"A": 3, "C": 1}, {"A": 0.80, "B": 0.10}) == pytest.approx(
        {"A": 0.05, "B": 0.10}
    )
    assert total_quota_deficit({}, quotas) == pytest.approx(0.80)
    assert quota_satisfaction_rate({}, quotas) == 0.0
    assert quota_satisfaction_rate({}, {"A": 0.0}) == 1.0
    assert total_quota_deficit({}, {"A": 0.0}) == 0.0
    assert quota_satisfaction_rate({"A": 1, "B": 1}, {"A": 0.5000000000005}) == 1.0
