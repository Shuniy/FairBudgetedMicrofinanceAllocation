import numpy as np
import pandas as pd
import pytest

from episode_statistics import describe, paired_comparisons


def test_student_interval_uses_sample_standard_deviation_and_valid_n():
    result = describe(pd.Series([10, 14, np.nan]))
    assert result == pytest.approx(dict(
        mean=12, std=2.8284271247461903, se=2, min=10, max=14, n=2,
        ci95_low=-13.41240947286419, ci95_high=37.41240947286419,
    ))


@pytest.mark.parametrize(("values", "limits"), [
    ([0] * 10, (0, 0.2775327998628892)),
    ([1] * 10, (0.7224672001371107, 1)),
    ([1, 0, 0], (0.0614919447203962, 0.7923403991979523)),
])
def test_wilson_interval_at_interior_and_boundaries(values, limits):
    result = describe(pd.Series(values), proportion=True)
    assert (result["ci95_low"], result["ci95_high"]) == pytest.approx(limits)
    assert result["mean"] == sum(values) / len(values)


def test_description_leaves_insufficient_statistics_undefined():
    one = describe(pd.Series([4, np.nan]))
    assert one["n"] == 1 and one["mean"] == 4
    assert all(np.isnan(one[field]) for field in ("std", "se", "ci95_low", "ci95_high"))
    empty = describe(pd.Series([np.nan]))
    assert empty["n"] == 0
    assert all(np.isnan(value) for field, value in empty.items() if field != "n")


def test_paired_differences_use_seed_keys_and_detect_missing_or_duplicate_episodes():
    frame = pd.DataFrame({
        "policy": ["fair_ucb_soft", "budgeted_ucb", "fair_ucb_soft", "budgeted_ucb"],
        "seed": [7, 8, 8, 7],
        "cumulative_reward": [11, 20, 23, 10],
        "reward_per_cost": [0.11, 0.20, 0.23, 0.10],
        "allocation_share_gap": [0.12, 0.20, 0.18, 0.11],
        "total_quota_deficit": [0, 0.20, 0, 0.11],
    })
    reward = paired_comparisons(frame, "arm", [7, 8]).set_index("metric").loc["cumulative_reward"]
    assert reward["mean"] == 2
    assert reward["std"] == pytest.approx(1.4142135623730951)
    assert (reward["ci95_low"], reward["ci95_high"]) == pytest.approx(
        (-10.706204736432095, 14.706204736432095)
    )
    with pytest.raises(ValueError, match="Missing"):
        paired_comparisons(frame.iloc[:-1], "arm", [7, 8])
    with pytest.raises(ValueError, match="Duplicate"):
        paired_comparisons(pd.concat([frame, frame.iloc[:1]]), "arm", [7, 8])
