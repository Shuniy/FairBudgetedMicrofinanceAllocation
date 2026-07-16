from __future__ import annotations

import pytest

from metrics import (
    allocation_shares,
    demographic_parity_gap,
    group_funding_rates,
    quota_deficits,
    quota_satisfaction_rate,
)


def test_metrics_handle_zero_and_missing_opportunities() -> None:
    rates = group_funding_rates({"A": 1, "B": 0}, {"A": 2, "B": 0})
    assert rates == {"A": 0.5, "B": 0.0}
    assert demographic_parity_gap({"A": 1, "B": 0}, {"A": 2, "B": 0}) == 0.5


def test_quota_metrics_use_allocation_shares() -> None:
    counts = {"A": 3, "B": 1}
    assert allocation_shares(counts) == {"A": 0.75, "B": 0.25}
    assert quota_deficits(counts, {"A": 0.5, "B": 0.5}) == {
        "A": 0.0,
        "B": 0.25,
    }
    assert quota_satisfaction_rate(counts, {"A": 0.5, "B": 0.5}) == pytest.approx(0.5)
