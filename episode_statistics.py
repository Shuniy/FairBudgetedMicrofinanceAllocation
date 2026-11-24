"""Statistics for independent seeds and within-configuration policy contrasts."""

from math import sqrt
from statistics import NormalDist

import numpy as np
import pandas as pd
from scipy.special import stdtrit


STATISTICS = ("mean", "std", "se", "min", "max", "n", "ci95_low", "ci95_high")
CI_METHODS = {
    "mean": "95% Student's t interval with n-1 degrees of freedom",
    "proportion": "95% Wilson score interval",
}
Z_95 = NormalDist().inv_cdf(0.975)


def describe(values: pd.Series, *, proportion: bool = False) -> dict[str, float | int]:
    values = values.dropna().to_numpy(dtype=float)
    n = len(values)
    if not np.isfinite(values).all() or (proportion and not np.isin(values, (0, 1)).all()):
        raise ValueError("Invalid episode values")
    mean = float(values.mean()) if n else np.nan
    std = float(values.std(ddof=1)) if n > 1 else np.nan
    se = std / sqrt(n) if n else np.nan
    low = high = np.nan
    if proportion and n:
        z_squared = Z_95 ** 2
        center = (mean + z_squared / (2 * n)) / (1 + z_squared / n)
        margin = Z_95 * sqrt(mean * (1 - mean) / n + z_squared / (4 * n * n))
        margin /= 1 + z_squared / n
        low, high = (0.0 if mean == 0 else center - margin,
                     1.0 if mean == 1 else center + margin)
    elif n > 1:
        margin = float(stdtrit(n - 1, 0.975)) * se
        low, high = mean - margin, mean + margin
    return dict(mean=mean, std=std, se=se, min=float(values.min()) if n else np.nan,
                max=float(values.max()) if n else np.nan, n=n, ci95_low=low, ci95_high=high)


# These are the existing fairness variants versus their corresponding unwrapped policies.
FAIRNESS_COMPARISONS = {
    "arm": {
        "fair_ucb_hard": "budgeted_ucb",
        "fair_ucb_soft": "budgeted_ucb",
    },
    "contextual": {
        f"fair_{family}_{variant}": f"budgeted_{base}"
        for family, base in (("epsilon_greedy", "contextual_epsilon_greedy"),
                             ("linucb", "linucb"), ("thompson", "contextual_thompson"))
        for variant in ("hard", "soft")
    },
    "cmdp": {
        f"cmdp_fair_{variant}_{family}": f"portfolio_{family}"
        for family in ("epsilon_greedy", "linucb", "thompson")
        for variant in ("hard", "lagrangian")
    },
}


def paired_comparisons(summaries: pd.DataFrame, mode: str, seeds: list[int]) -> pd.DataFrame:
    """Compare policies from one run, whose manifest supplies configuration and seed list."""
    expected = pd.Index(sorted(seeds), name="seed")
    if expected.has_duplicates:
        raise ValueError("Duplicate seeds in requested seed list")
    results = []
    parity = "allocation_share_gap" if mode == "arm" else "feasible_dp_gap"
    metrics = ("cumulative_reward", "reward_per_cost", parity, "total_quota_deficit")
    for policy, baseline in FAIRNESS_COMPARISONS[mode].items():
        if policy not in summaries["policy"].values:
            continue
        episodes = []
        for name in (policy, baseline):
            frame = summaries.loc[summaries["policy"].eq(name)].set_index("seed").sort_index()
            if frame.index.has_duplicates:
                raise ValueError(f"Duplicate paired seed for {name}")
            if not frame.index.equals(expected):
                raise ValueError(f"Missing or unexpected paired seed for {name}")
            episodes.append(frame)
        for metric in metrics:
            differences = episodes[0][metric] - episodes[1][metric]
            results.append({"policy": policy, "baseline": baseline, "metric": metric,
                            **describe(differences)})
    return pd.DataFrame(results, columns=["policy", "baseline", "metric", *STATISTICS])
