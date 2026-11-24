from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from html import escape
from math import ceil, floor, sqrt
from os.path import relpath
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Data.feature_encoder import BorrowerFeatureEncoder
from evaluate import (
    CONSTRAINT_RATE_FIELDS,
    FAIRNESS_COUNT_PREFIXES,
    REPORT_STYLE,
    CMDPPolicyFactory,
    compare_cmdp_policies,
    reproducibility_metadata,
)
from metrics import QUOTA_NUMERICAL_TOLERANCE, total_quota_deficit
from models.environment_config import CMDPEnvironmentConfig


BUDGET_SWEEP_VALUES = (0.50, 0.75, 1.00, 1.25, 1.50)
IMBALANCE_SWEEP_VALUES = (0.10, 0.20, 0.30, 0.40, 0.50)
FAIRNESS_SWEEP_VALUES = (0.00, 0.25, 0.50, 0.75, 1.00, 1.25)

QUICK_BUDGET_SWEEP_VALUES = (0.50, 1.00)
QUICK_IMBALANCE_SWEEP_VALUES = (0.10, 0.20)
QUICK_FAIRNESS_SWEEP_VALUES = (0.00, 1.00)


@dataclass(frozen=True, slots=True)
class CMDPSweepCase:
    """One environment/policy configuration in a Phase 3 robustness sweep."""

    value: float
    env_config: CMDPEnvironmentConfig
    policy_factories: Mapping[str, CMDPPolicyFactory]
    target_shares: Mapping[str, float]
    parity_tolerance: float
    parameter_kind: str
    parameter_details: Mapping[str, object]
    is_baseline: bool = False


@dataclass(frozen=True, slots=True)
class SweepArtifacts:
    name: str
    output: Path
    raw_results: pd.DataFrame
    aggregate_results: pd.DataFrame
    summary_statements: tuple[str, ...]
    plot_paths: tuple[Path, ...]


def _quota_opportunity_feasible(
    opportunities: Mapping[str, int],
    loans_funded: int,
    target_shares: Mapping[str, float],
    tolerance: float = 0.0,
) -> bool:
    """Screen achievable total deficit at the observed integer loan volume."""

    if loans_funded > sum(opportunities.values()):
        return False
    if loans_funded == 0:
        return total_quota_deficit({}, target_shares) <= tolerance

    # A loan below a group's target reduces deficit by one loan unit.
    # At most one additional loan per group removes a fractional unit.
    whole_units = 0
    fractional_units = []
    total_target_share = 0.0
    for group, target in target_shares.items():
        if target <= 0:
            continue
        total_target_share += target
        available = opportunities.get(group, 0)
        target_units = target * loans_funded
        whole = floor(target_units)
        whole_units += min(whole, available)
        if available > whole:
            fractional_units.append(target_units - whole)

    remaining = max(0, loans_funded - whole_units)
    covered = min(loans_funded, whole_units) + sum(
        sorted(fractional_units, reverse=True)[:remaining]
    )
    minimum_deficit = max(0.0, total_target_share - covered / loans_funded)
    return minimum_deficit <= tolerance + QUOTA_NUMERICAL_TOLERANCE


def _parity_opportunity_feasible(
    opportunities: Mapping[str, int],
    loans_funded: int,
    tolerance: float,
) -> bool | None:
    """Integer screen for parity feasibility at the observed aggregate volume.

    It asks whether integer allocations with the actual total loan count could
    achieve funding rates within ``tolerance`` using the observed opportunity
    counts. It is deliberately an opportunity-count screen, not a claim that
    timing, liquidity, and per-round batch constraints are jointly feasible.
    """

    counts = [int(value) for value in opportunities.values() if value > 0]
    if loans_funded > sum(counts):
        return False
    if len(counts) < 2:
        return None
    if loans_funded == 0:
        return True
    candidate_minimum_rates = {0.0}
    for count in counts:
        for allocations in range(min(count, loans_funded) + 1):
            candidate_minimum_rates.add(allocations / count)
    for minimum_rate in candidate_minimum_rates:
        lower = [
            ceil(max(0.0, minimum_rate * count - 1e-12))
            for count in counts
        ]
        upper = [
            floor(min(1.0, minimum_rate + tolerance) * count + 1e-12)
            for count in counts
        ]
        if all(lo <= hi for lo, hi in zip(lower, upper)) and (
            sum(lower) <= loans_funded <= sum(upper)
        ):
            return True
    return False


def _episode_opportunities(row: pd.Series) -> dict[str, int]:
    return {
        column.removeprefix("opportunities_"): int(value)
        for column, value in row.items()
        if column.startswith("opportunities_") and pd.notna(value)
    }


def _add_sweep_metrics(
    summaries: pd.DataFrame,
    trajectories: pd.DataFrame,
    case: CMDPSweepCase,
    sweep_name: str,
    quota_tolerance: float,
) -> pd.DataFrame:
    raw = summaries.copy()
    raw["sweep_name"] = sweep_name
    raw["sweep_value"] = float(case.value)
    raw["parameter_kind"] = case.parameter_kind
    raw["is_baseline"] = bool(case.is_baseline)
    raw["initial_budget"] = case.env_config.initial_capital
    raw["eventual_cumulative_reward"] = raw["cumulative_reward"]
    raw["cumulative_immediate_reward"] = raw["observed_reward_by_horizon"]
    raw["cumulative_cost"] = raw["cumulative_disbursed"]
    raw["recycled_capital"] = raw["cumulative_recycled"]
    raw["final_remaining_capital"] = raw["final_available_capital"]
    raw["rounds_completed"] = raw["rounds"]
    raw["budget_utilization"] = (
        raw["initial_budget"] - raw["final_available_capital"]
    ) / raw["initial_budget"]
    raw["parity_excess"] = np.maximum(
        0.0,
        raw["demographic_parity_gap"] - case.parity_tolerance,
    )
    raw["quota_excess"] = np.maximum(
        0.0,
        raw["total_quota_deficit"] - quota_tolerance,
    )
    raw["constraint_violation"] = raw["parity_excess"] + raw["quota_excess"]
    raw["fairness_violation"] = raw["constraint_violation"]

    if trajectories.empty:
        raw["fairness_violations"] = 0
    else:
        trajectory_violations = trajectories.assign(
            _violated=(
                trajectories["parity_satisfied"].eq(0)
                | ~trajectories["quota_satisfied"]
            ).astype(int)
        )
        counts = trajectory_violations.groupby(["policy", "seed"])[
            "_violated"
        ].sum()
        raw["fairness_violations"] = [
            int(counts.get((policy, seed), 0))
            for policy, seed in zip(raw["policy"], raw["seed"])
        ]

    quota_feasible: list[bool] = []
    parity_feasible: list[bool | None] = []
    for _, row in raw.iterrows():
        opportunities = _episode_opportunities(row)
        loans_funded = int(row["loans_funded"])
        quota_feasible.append(
            _quota_opportunity_feasible(
                opportunities,
                loans_funded,
                case.target_shares,
                quota_tolerance,
            )
        )
        parity_feasible.append(
            _parity_opportunity_feasible(
                opportunities,
                loans_funded,
                case.parity_tolerance,
            )
        )
    raw["quota_feasible"] = quota_feasible
    raw["parity_feasible"] = pd.Series(
        parity_feasible, index=raw.index, dtype="boolean"
    )
    raw["opportunity_screens_passed"] = raw["quota_feasible"] & raw["parity_feasible"]
    raw["fairness_feasible"] = raw["opportunity_screens_passed"]  # Legacy screen alias.
    raw["feasibility_status"] = np.select(
        [
            raw["opportunity_screens_passed"].eq(False).to_numpy(dtype=bool, na_value=False),
            ~raw["parity_evaluable"],
            raw["constraint_satisfied"].eq(1),
        ],
        ["opportunity_infeasible", "parity_not_evaluable", "constraint_satisfied"],
        default="policy_constraint_violation",
    )

    for key, value in case.parameter_details.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            raw[key] = value

    group_columns = [
        column
        for column in raw.columns
        if column.startswith(
            (
                *FAIRNESS_COUNT_PREFIXES,
                "allocation_share_",
                "quota_deficit_",
            )
        )
    ]
    raw[group_columns] = raw[group_columns].fillna(0)
    opportunity_columns = [
        column for column in raw if column.startswith("opportunities_")
    ]
    raw["total_opportunities"] = raw[opportunity_columns].sum(axis=1)
    if "opportunities_minority" not in raw:
        raw["opportunities_minority"] = 0
    if "funded_minority" not in raw:
        raw["funded_minority"] = 0
    raw["observed_primary_minority_share"] = np.where(
        raw["total_opportunities"] > 0,
        raw["opportunities_minority"] / raw["total_opportunities"],
        0.0,
    )
    population_columns = [
        column for column in raw if column.startswith("population_opportunities_")
    ]
    raw["total_population_opportunities"] = raw[population_columns].sum(axis=1)
    raw["observed_population_primary_minority_share"] = (
        raw.get("population_opportunities_minority", 0)
        / raw["total_population_opportunities"].replace(0, np.nan)
    )
    return raw


AGGREGATE_METRICS = (
    "eventual_cumulative_reward",
    "cumulative_immediate_reward",
    "cumulative_cost",
    "reward_per_cost",
    "repayment_rate",
    "budget_utilization",
    "loans_funded",
    "rounds_completed",
    "cumulative_recycled",
    "default_loss_by_horizon",
    "unrecycled_successful_principal_by_horizon",
    "recycled_capital",
    "final_available_capital",
    "final_remaining_capital",
    "final_capital_at_risk",
    "eventual_terminal_capital",
    "capital_turnover",
    "capital_survival_ratio",
    "demographic_parity_gap",
    "feasible_dp_gap",
    "population_dp_gap",
    "feasible_dp_groups",
    "population_dp_groups",
    "quota_satisfaction_rate",
    "total_quota_deficit",
    "fairness_violations",
    "constraint_violation",
    "quota_feasible",
    "parity_feasible",
    "fairness_feasible",
    "opportunity_screens_passed",
    "lambda_quota_final",
    "lambda_parity_final",
    "funded_minority",
    "opportunities_minority",
    "funding_rate_minority",
    "observed_primary_minority_share",
    "observed_population_primary_minority_share",
    "total_opportunities",
    "total_population_opportunities",
)


def aggregate_sweep_results(raw_results: pd.DataFrame) -> pd.DataFrame:
    """Return tidy seed aggregates with normal-approximation 95% intervals."""

    group_columns = [
        "sweep_name",
        "sweep_value",
        "parameter_kind",
        "policy",
        "is_baseline",
    ]
    metrics = {metric: metric for metric in AGGREGATE_METRICS if metric in raw_results}
    metrics.update(
        (column, column) for column in raw_results
        if column.startswith(FAIRNESS_COUNT_PREFIXES)
    )
    metrics.update(
        (rate, outcome) for rate, outcome in CONSTRAINT_RATE_FIELDS.items()
        if outcome in raw_results
    )
    records: list[dict[str, object]] = []
    for keys, frame in raw_results.groupby(group_columns, dropna=False, sort=True):
        identity = dict(zip(group_columns, keys))
        identity["episodes_evaluated"] = len(frame)
        for field in ("parity_evaluable", "population_parity_evaluable"):
            if field in frame:
                identity[f"{field}_episodes"] = int(frame[field].sum())
        for metric, source in metrics.items():
            values = pd.to_numeric(frame[source], errors="coerce")
            if (
                source in CONSTRAINT_RATE_FIELDS.values()
                or source.startswith(FAIRNESS_COUNT_PREFIXES)
            ):
                values = values.fillna(0)
            values = values.dropna()
            n_seeds = int(len(values))
            mean = float(values.mean()) if n_seeds else np.nan
            std = float(values.std(ddof=1)) if n_seeds > 1 else 0.0
            margin = 1.96 * std / sqrt(n_seeds) if n_seeds else 0.0
            records.append(
                {
                    **identity,
                    "metric": metric,
                    "metric_mean": mean,
                    "metric_std": std,
                    "metric_ci95_low": mean - margin,
                    "metric_ci95_high": mean + margin,
                    "n_seeds": n_seeds,
                }
            )
    return pd.DataFrame(records)


def _metric_rows(aggregate: pd.DataFrame, metric: str) -> pd.DataFrame:
    return aggregate.loc[
        aggregate["metric"].eq(metric) & aggregate["metric_mean"].notna()
    ].copy()


def _plot_metric_by_value(
    aggregate: pd.DataFrame,
    metric: str,
    output_path: Path,
    x_label: str,
    title: str,
    baseline_value: float,
) -> bool:
    frame = _metric_rows(aggregate, metric)
    if frame.empty:
        return False
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for policy, policy_frame in frame.groupby("policy", sort=True):
        policy_frame = policy_frame.sort_values("sweep_value")
        x = policy_frame["sweep_value"].to_numpy(dtype=float)
        mean = policy_frame["metric_mean"].to_numpy(dtype=float)
        low = policy_frame["metric_ci95_low"].to_numpy(dtype=float)
        high = policy_frame["metric_ci95_high"].to_numpy(dtype=float)
        (line,) = ax.plot(x, mean, marker="o", linewidth=1.5, label=policy)
        ax.fill_between(x, low, high, color=line.get_color(), alpha=0.10)
    ax.axvline(
        baseline_value,
        color="#111827",
        linestyle="--",
        linewidth=1.0,
        label="Phase 3 default",
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return True


def _plot_frontier(
    aggregate: pd.DataFrame,
    x_metric: str,
    y_metric: str,
    output_path: Path,
    title: str,
) -> bool:
    x_frame = _metric_rows(aggregate, x_metric)
    y_frame = _metric_rows(aggregate, y_metric)
    identity = [
        "sweep_name",
        "sweep_value",
        "parameter_kind",
        "policy",
        "is_baseline",
    ]
    merged = x_frame.merge(y_frame, on=identity, suffixes=("_x", "_y"))
    if merged.empty:
        return False
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for policy, frame in merged.groupby("policy", sort=True):
        frame = frame.sort_values("sweep_value")
        ax.plot(
            frame["metric_mean_x"],
            frame["metric_mean_y"],
            marker="o",
            linewidth=1.5,
            label=policy,
        )
        for row in frame.itertuples(index=False):
            ax.annotate(
                f"{row.sweep_value:g}",
                (row.metric_mean_x, row.metric_mean_y),
                fontsize=6,
                xytext=(3, 3),
                textcoords="offset points",
            )
    ax.set_xlabel(x_metric.replace("_", " ").title())
    ax.set_ylabel(y_metric.replace("_", " ").title())
    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return True


def _average_extreme_statement(
    aggregate: pd.DataFrame,
    metric: str,
    highest: bool,
    setting_label: str,
) -> str | None:
    frame = _metric_rows(aggregate, metric)
    if frame.empty:
        return None
    averages = frame.groupby("policy")["metric_mean"].mean()
    policy = averages.idxmax() if highest else averages.idxmin()
    criterion = "highest" if highest else "lowest"
    return (
        f"Across the tested {setting_label}, {policy} had the {criterion} "
        f"average {metric.replace('_', ' ')} ({averages.loc[policy]:.4f})."
    )


def build_sweep_summary(
    sweep_name: str,
    aggregate: pd.DataFrame,
) -> tuple[str, ...]:
    statements: list[str] = []
    setting_label = {
        "budget": "budget settings",
        "group_imbalance": "group-imbalance settings",
        "fairness_strength": "fairness-strength settings",
    }[sweep_name]
    reward = _average_extreme_statement(
        aggregate,
        "eventual_cumulative_reward",
        True,
        setting_label,
    )
    parity = _average_extreme_statement(
        aggregate,
        "feasible_dp_gap",
        False,
        setting_label,
    )
    statements.extend(statement for statement in (reward, parity) if statement)

    if sweep_name == "budget":
        reward_rows = _metric_rows(aggregate, "eventual_cumulative_reward")
        parity_rows = _metric_rows(aggregate, "feasible_dp_gap")
        baseline = reward_rows.loc[reward_rows["is_baseline"]]
        tight_value = float(reward_rows["sweep_value"].min())
        if not baseline.empty:
            reference_policy = baseline.loc[
                baseline["metric_mean"].idxmax(), "policy"
            ]
            baseline_reward = float(
                baseline.loc[
                    baseline["policy"] == reference_policy,
                    "metric_mean",
                ].iloc[0]
            )
            tight_reward_rows = reward_rows.loc[
                (reward_rows["policy"] == reference_policy)
                & np.isclose(reward_rows["sweep_value"], tight_value)
            ]
            baseline_gap_rows = parity_rows.loc[
                (parity_rows["policy"] == reference_policy)
                & parity_rows["is_baseline"]
            ]
            tight_gap_rows = parity_rows.loc[
                (parity_rows["policy"] == reference_policy)
                & np.isclose(parity_rows["sweep_value"], tight_value)
            ]
            if (
                not tight_reward_rows.empty
                and not baseline_gap_rows.empty
                and not tight_gap_rows.empty
            ):
                tight_reward = float(tight_reward_rows.iloc[0]["metric_mean"])
                reward_change = (
                    100.0 * (tight_reward - baseline_reward) / baseline_reward
                    if abs(baseline_reward) > 1e-12
                    else 0.0
                )
                baseline_gap = float(baseline_gap_rows.iloc[0]["metric_mean"])
                tight_gap = float(tight_gap_rows.iloc[0]["metric_mean"])
                statements.append(
                    f"For the default-setting reward leader {reference_policy}, "
                    f"reducing initial capital to {tight_value:.0%} changed "
                    f"eventual reward by {reward_change:+.2f}% and feasible parity gap "
                    f"by {tight_gap - baseline_gap:+.4f}."
                )
        quota = _metric_rows(aggregate, "quota_satisfaction_rate")
        for policy, frame in quota.groupby("policy"):
            if "fair_hard" not in policy:
                continue
            maintained = int((frame["metric_mean"] >= 1.0 - 1e-12).sum())
            statements.append(
                f"{policy} maintained mean quota satisfaction of 1.0 in "
                f"{maintained} of {len(frame)} budget settings."
            )
    elif sweep_name == "group_imbalance":
        satisfied = _metric_rows(aggregate, "constraint_satisfaction_rate")
        for policy, frame in satisfied.groupby("policy"):
            if "fair_hard" not in policy:
                continue
            settings = frame.loc[
                frame["metric_mean"] >= 1.0 - 1e-12,
                "sweep_value",
            ]
            if settings.empty:
                statements.append(
                    f"{policy} did not satisfy both final fairness thresholds "
                    "for every seed at any tested minority-share setting."
                )
            else:
                statements.append(
                    f"{policy} satisfied both final fairness thresholds for "
                    f"every seed down to configured minority share "
                    f"{settings.min():.0%}."
                )
    elif sweep_name == "fairness_strength":
        reward_rows = _metric_rows(aggregate, "eventual_cumulative_reward")
        gap_rows = _metric_rows(aggregate, "feasible_dp_gap")
        identity = ["policy", "sweep_value", "parameter_kind"]
        frontier = reward_rows.merge(
            gap_rows,
            on=identity,
            suffixes=("_reward", "_gap"),
        )
        for policy, frame in frontier.groupby("policy"):
            best_gap = frame.loc[frame["metric_mean_gap"].idxmin()]
            reference = frame.loc[np.isclose(frame["sweep_value"], 0.0)]
            if reference.empty:
                continue
            reference_reward = float(reference.iloc[0]["metric_mean_reward"])
            reward_at_gap = float(best_gap["metric_mean_reward"])
            reduction = (
                100.0 * (reference_reward - reward_at_gap) / reference_reward
                if abs(reference_reward) > 1e-12
                else 0.0
            )
            statements.append(
                f"For {policy}, the lowest mean feasible parity gap occurred at strength "
                f"{float(best_gap['sweep_value']):g} "
                f"({float(best_gap['metric_mean_gap']):.4f}); eventual reward "
                f"changed by {-reduction:+.2f}% relative to strength 0."
            )
    return tuple(statements)


BASELINE_REQUIRED_FAIRNESS_FIELDS = (
    "feasible_dp_gap",
    "population_dp_gap",
    "feasible_dp_groups",
    "population_dp_groups",
    "parity_evaluable",
    "population_parity_evaluable",
    *CONSTRAINT_RATE_FIELDS.values(),
)


BASELINE_REQUIRED_ACCOUNTING_FIELDS = (
    "default_loss_by_horizon",
    "unrecycled_successful_principal_by_horizon",
    "pending_loans",
    "capital_survival_ratio",
)


BASELINE_COMPARISON_METRICS = (
    "loans_funded",
    "cumulative_reward",
    "observed_reward_by_horizon",
    "repayment_rate",
    "cumulative_disbursed",
    "reward_per_cost",
    "cumulative_recycled",
    "final_available_capital",
    "final_capital_at_risk",
    "eventual_terminal_capital",
    *BASELINE_REQUIRED_ACCOUNTING_FIELDS,
    "demographic_parity_gap",
    *BASELINE_REQUIRED_FAIRNESS_FIELDS,
    "quota_satisfaction_rate",
    "total_quota_deficit",
    "lambda_quota_final",
    "lambda_parity_final",
)


def _baseline_frames_match(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
) -> tuple[bool, list[str]]:
    keys = ["policy", "seed"]
    reference = reference.set_index(keys).sort_index()
    candidate = candidate.set_index(keys).sort_index()
    errors: list[str] = []
    for label, frame in (("reference", reference), ("candidate", candidate)):
        if frame.index.has_duplicates:
            errors.append(f"duplicate policy/seed rows in {label}")
        missing = [field for field in BASELINE_REQUIRED_FAIRNESS_FIELDS if field not in frame]
        if missing:
            errors.append(f"{label} missing required fairness fields: {', '.join(missing)}")
        missing = [field for field in BASELINE_REQUIRED_ACCOUNTING_FIELDS if field not in frame]
        if missing:
            errors.append(f"{label} missing required accounting fields: {', '.join(missing)}")
    if not reference.index.equals(candidate.index):
        errors.append("policy/seed rows differ")
    if errors:
        return False, errors

    metrics = [
        metric
        for metric in BASELINE_COMPARISON_METRICS
        if metric in reference and metric in candidate
    ]
    count_columns = [
        column for column in reference.columns.union(candidate.columns)
        if column.startswith(FAIRNESS_COUNT_PREFIXES)
    ]
    metrics.extend(count_columns)
    reference = reference.reindex(columns=metrics, fill_value=0)
    candidate = candidate.reindex(columns=metrics, fill_value=0)
    for metric in metrics:
        left, right = reference[metric], candidate[metric]
        if metric in count_columns:
            left, right = left.fillna(0), right.fillna(0)
        left = pd.to_numeric(left, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
        right = pd.to_numeric(right, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
        if not np.allclose(left, right, rtol=0.0, atol=1e-12, equal_nan=True):
            errors.append(metric)
    return not errors, errors


def validate_mode4_baselines(
    artifacts: Mapping[str, SweepArtifacts],
    output: Path,
) -> dict[str, object]:
    """Compare Mode 4 sweep defaults for matching policy and seed rows."""

    budget = artifacts["budget"].raw_results
    imbalance = artifacts["group_imbalance"].raw_results
    fairness = artifacts["fairness_strength"].raw_results
    budget_default = budget.loc[budget["is_baseline"]]
    imbalance_default = imbalance.loc[imbalance["is_baseline"]]
    fair_default = fairness.loc[fairness["is_baseline"]]
    fair_policies = set(fair_default["policy"])
    budget_fair_default = budget_default.loc[
        budget_default["policy"].isin(fair_policies)
    ]
    group_match, group_errors = _baseline_frames_match(
        budget_default,
        imbalance_default,
    )
    fairness_match, fairness_errors = _baseline_frames_match(
        budget_fair_default,
        fair_default,
    )
    result: dict[str, object] = {
        "budget_vs_group_imbalance_default": group_match,
        "budget_vs_fairness_strength_default": fairness_match,
        "all_passed": group_match and fairness_match,
        "differences": {
            "group_imbalance": group_errors,
            "fairness_strength": fairness_errors,
        },
    }
    (output / "baseline_consistency.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not result["all_passed"]:
        raise RuntimeError(
            "Mode 4 sweep default points differ: "
            f"{result['differences']}"
        )
    return result


def _plot_sweep(
    sweep_name: str,
    aggregate: pd.DataFrame,
    plots_dir: Path,
    baseline_value: float,
) -> tuple[Path, ...]:
    plots_dir.mkdir(parents=True, exist_ok=True)
    x_labels = {
        "budget": "Budget multiplier",
        "group_imbalance": "Configured marginalized-community probability",
        "fairness_strength": "Policy-specific fairness strength",
    }
    metrics = [
        "eventual_cumulative_reward",
        "feasible_dp_gap",
        "population_dp_gap",
        "quota_satisfaction_rate",
    ]
    if sweep_name == "budget":
        metrics.append("reward_per_cost")
    elif sweep_name == "group_imbalance":
        metrics.append("funding_rate_minority")
    else:
        metrics.append("budget_utilization")
    paths: list[Path] = []
    for metric in metrics:
        path = plots_dir / f"{metric}.png"
        if _plot_metric_by_value(
            aggregate,
            metric,
            path,
            x_labels[sweep_name],
            f"{sweep_name.replace('_', ' ').title()}: {metric.replace('_', ' ').title()}",
            baseline_value,
        ):
            paths.append(path)

    frontier_metrics = [("feasible_dp_gap", "eventual_cumulative_reward")]
    if sweep_name == "fairness_strength":
        frontier_metrics.append(
            ("quota_satisfaction_rate", "eventual_cumulative_reward")
        )
    for x_metric, y_metric in frontier_metrics:
        path = plots_dir / f"{y_metric}_vs_{x_metric}.png"
        if _plot_frontier(
            aggregate,
            x_metric,
            y_metric,
            path,
            (
                f"{sweep_name.replace('_', ' ').title()}: "
                f"{y_metric.replace('_', ' ').title()} vs "
                f"{x_metric.replace('_', ' ').title()}"
            ),
        ):
            paths.append(path)
    return tuple(paths)


def run_cmdp_sweep(
    sweep_name: str,
    cases: Sequence[CMDPSweepCase],
    seeds: Sequence[int],
    output: Path,
    default_config: CMDPEnvironmentConfig,
    quota_tolerance: float,
    baseline_value: float,
    encoder: BorrowerFeatureEncoder | None = None,
) -> SweepArtifacts:
    """Run and persist a generic Phase 3-only sensitivity sweep."""

    if not cases:
        raise ValueError("a CMDP sweep requires at least one case")
    if not seeds:
        raise ValueError("a CMDP sweep requires at least one seed")
    shared_encoder = encoder or BorrowerFeatureEncoder(
        include_protected_attributes=False
    )
    raw_frames: list[pd.DataFrame] = []
    trajectory_frames: list[pd.DataFrame] = []
    funding_frames: list[pd.DataFrame] = []
    maturity_frames: list[pd.DataFrame] = []
    case_policy_configurations = []

    for case in cases:
        summaries, trajectories, funding, maturities = compare_cmdp_policies(
            env_config=case.env_config,
            policy_factories=case.policy_factories,
            target_shares=case.target_shares,
            parity_tolerance=case.parity_tolerance,
            seeds=seeds,
            encoder=shared_encoder,
            quota_tolerance=quota_tolerance,
        )
        case_policy_configurations.append(summaries.attrs["policy_configurations"])
        raw_frames.append(
            _add_sweep_metrics(
                summaries,
                trajectories,
                case,
                sweep_name,
                quota_tolerance,
            )
        )
        for frame, collection in (
            (trajectories, trajectory_frames),
            (funding, funding_frames),
            (maturities, maturity_frames),
        ):
            copied = frame.copy()
            copied["sweep_name"] = sweep_name
            copied["sweep_value"] = case.value
            copied["parameter_kind"] = case.parameter_kind
            copied["is_baseline"] = case.is_baseline
            collection.append(copied)

    raw = pd.concat(raw_frames, ignore_index=True, sort=False)
    aggregate = aggregate_sweep_results(raw)
    output.mkdir(parents=True, exist_ok=True)
    raw.to_csv(output / "raw_results.csv", index=False)
    # Retire the duplicate filename when rerunning an older sweep directory.
    (output / "episode_summaries.csv").unlink(missing_ok=True)
    aggregate.to_csv(output / "aggregate_results.csv", index=False)
    for frames, filename in (
        (trajectory_frames, "trajectories.csv"),
        (funding_frames, "funding_log.csv"),
        (maturity_frames, "maturity_log.csv"),
    ):
        combined = pd.concat(frames, ignore_index=True, sort=False)
        combined.to_csv(output / filename, index=False)

    manifest = {
        "sweep_name": sweep_name,
        "sweep_variable": {
            "budget": "initial_capital multiplier",
            "group_imbalance": "marginalized_probability",
            "fairness_strength": "policy-specific fairness strength",
        }[sweep_name],
        "values": sorted({float(case.value) for case in cases}),
        "baseline_value": baseline_value,
        "seeds": list(seeds),
        "policies": sorted(
            {
                policy
                for case in cases
                for policy in case.policy_factories
            }
        ),
        "horizon": default_config.max_rounds,
        "quota_tolerance": quota_tolerance,
        "default_phase3_parameters": asdict(default_config),
        "protected_attributes_in_prediction": shared_encoder.include_protected_attributes,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **reproducibility_metadata(),
        "cases": [
            {
                "value": case.value,
                "parameter_kind": case.parameter_kind,
                "is_baseline": case.is_baseline,
                "environment": asdict(case.env_config),
                "target_shares": dict(case.target_shares),
                "parity_tolerance": case.parity_tolerance,
                "parameter_details": dict(case.parameter_details),
                "policies": list(case.policy_factories),
                "policy_configurations": policy_configurations,
            }
            for case, policy_configurations in zip(cases, case_policy_configurations)
        ],
        "feasibility_definition": (
            "opportunity_screens_passed combines separate quota and parity screens "
            "using feasible opportunities and the actual funded volume. Passing "
            "both leaves joint allocation feasibility unresolved, including costs, "
            "capital, batch limits, repayment delays and arrival timing. "
            "constraint_satisfied records the observed episode outcome."
        ),
    }
    (output / "config.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    statements = list(build_sweep_summary(sweep_name, aggregate))
    infeasible = int((raw["feasibility_status"] == "opportunity_infeasible").sum())
    not_evaluable = int((~raw["parity_evaluable"]).sum())
    policy_violations = int(
        (raw["feasibility_status"] == "policy_constraint_violation").sum()
    )
    constraints_satisfied = int(
        (raw["feasibility_status"] == "constraint_satisfied").sum()
    )
    statements.append(
        f"Feasible-opportunity screens ruled out satisfaction in {infeasible} of "
        f"{len(raw)} episodes. {policy_violations} episodes passed both screens "
        f"and violated a constraint. {constraints_satisfied} satisfied both "
        f"constraints. Feasible parity was undefined in {not_evaluable} episodes."
    )
    summary_payload = {
        "statements": statements,
        "episodes": int(len(raw)),
        "opportunity_infeasible_episodes": infeasible,
        "policy_constraint_violation_episodes": policy_violations,
        "constraint_satisfied_episodes": constraints_satisfied,
        "parity_not_evaluable_episodes": not_evaluable,
        "population_parity_not_evaluable_episodes": int(
            (~raw["population_parity_evaluable"]).sum()
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    plots = _plot_sweep(
        sweep_name,
        aggregate,
        output / "plots",
        baseline_value,
    )
    return SweepArtifacts(
        name=sweep_name,
        output=output,
        raw_results=raw,
        aggregate_results=aggregate,
        summary_statements=tuple(statements),
        plot_paths=plots,
    )


def _summary_table(artifact: SweepArtifacts) -> str:
    metrics = [
        "eventual_cumulative_reward",
        "feasible_dp_gap",
        "population_dp_gap",
        "quota_satisfaction_rate",
        "reward_per_cost",
        *CONSTRAINT_RATE_FIELDS,
    ]
    if artifact.name == "group_imbalance":
        metrics.append("funding_rate_minority")
    frame = artifact.aggregate_results.loc[
        artifact.aggregate_results["metric"].isin(metrics)
    ].copy()
    if frame.empty:
        return '<p class="note">No aggregate rows were produced.</p>'
    frame["estimate"] = frame.apply(
        lambda row: (
            f"{float(row['metric_mean']):.5f} "
            f"[{float(row['metric_ci95_low']):.5f}, "
            f"{float(row['metric_ci95_high']):.5f}]"
        ),
        axis=1,
    )
    display = frame.pivot_table(
        index=["sweep_value", "parameter_kind", "policy", "is_baseline"],
        columns="metric",
        values="estimate",
        aggfunc="first",
    ).reset_index()
    display = display.rename(
        columns={
            "sweep_value": "Sweep value",
            "parameter_kind": "Parameter",
            "policy": "Policy",
            "is_baseline": "Default",
            "eventual_cumulative_reward": "Eventual reward mean [95% CI]",
            "feasible_dp_gap": "Feasible DP gap mean [95% CI]",
            "population_dp_gap": "Population DP gap mean [95% CI]",
            "quota_satisfaction_rate": "Quota satisfaction mean [95% CI]",
            "reward_per_cost": "Reward/cost mean [95% CI]",
            "constraint_satisfaction_rate": "Feasible joint satisfaction [95% CI]",
            "population_constraint_satisfaction_rate": "Population joint satisfaction [95% CI]",
            "funding_rate_minority": "Minority funding rate mean [95% CI]",
        }
    )
    display["Default"] = display["Default"].map({True: "Yes", False: "No"})
    table = display.to_html(index=False, border=0, classes="results", escape=True)
    return f'<div class="table-scroll">{table}</div>'


def save_mode4_report(
    artifacts: Mapping[str, SweepArtifacts],
    output: Path,
    seeds: Sequence[int],
    quick_test: bool,
    baseline_consistency: Mapping[str, object],
) -> Path:
    """Write the existing-style HTML report with a dedicated Mode 4 section."""

    labels = {
        "budget": "Budget Sensitivity",
        "group_imbalance": "Group-Imbalance Sensitivity",
        "fairness_strength": "Fairness-Strength Ablation",
    }
    descriptions = {
        "budget": (
            "Only initial Phase 3 capital changes. Delayed repayment, recycling, "
            "population, horizon, fairness targets, and policies remain fixed. "
            "Budget utilisation is the fraction of initial capital that is not "
            "liquid at the decision horizon; cumulative deployment and capital "
            "turnover remain separate metrics."
        ),
        "group_imbalance": (
            "The sweep value is the generator probability that an applicant has "
            "marginalized_community=True. The audited primary 'minority' label can "
            "be lower because disability and reserved status have higher primary-"
            "group precedence. The repayment equation is unchanged."
        ),
        "fairness_strength": (
            "Hard policies scale quota targets while retaining the configured "
            "demographic-parity tolerance. Lagrangian policies scale the penalty "
            "applied to their adaptive dual variables while retaining quota "
            "targets and dual updates. These are separate mechanisms and are not "
            "interpreted as mathematically equivalent. Hard-policy quota "
            "satisfaction is evaluated against the scaled targets, while "
            "Lagrangian quota satisfaction uses the unchanged Phase 3 targets."
        ),
    }
    sections: list[str] = []
    for name in ("budget", "group_imbalance", "fairness_strength"):
        artifact = artifacts[name]
        values = sorted(artifact.raw_results["sweep_value"].unique())
        summary_items = "".join(
            f"<li>{escape(statement)}</li>"
            for statement in artifact.summary_statements
        )
        images = "".join(
            (
                f'<figure><img src="{escape(path.relative_to(output).as_posix())}" '
                f'alt="{escape(path.stem.replace("_", " "))}">'
                f"<figcaption>{escape(path.stem.replace('_', ' ').title())}</figcaption>"
                "</figure>"
            )
            for path in artifact.plot_paths
        )
        sections.append(
            f"<h2>{escape(labels[name])}</h2>"
            f"<p>{escape(descriptions[name])}</p>"
            f"<p><strong>Tested values:</strong> "
            f"{escape(', '.join(f'{value:g}' for value in values))}</p>"
            f"<h3>Data-derived summary</h3><ul>{summary_items}</ul>"
            f"<h3>Aggregate results</h3><p class=\"note\">Values are seed "
            "means with normal-approximation 95% confidence intervals. "
            "Constraint satisfaction rates use all episodes. The aggregate CSV "
            "records the number of episodes with evaluable parity for each "
            "denominator. Default marks the existing Phase 3 setting.</p>"
            f"{_summary_table(artifact)}<h3>Plots</h3>"
            f'<div class="plots">{images}</div>'
        )

    manifest = {
        "mode": "Phase 3 CMDP robustness sweeps",
        "quick_test": quick_test,
        "seeds": list(seeds),
        "sweeps": {
            name: sorted(artifact.raw_results["sweep_value"].unique().tolist())
            for name, artifact in artifacts.items()
        },
        "sweep_configurations": {
            name: relpath(artifact.output / "config.json", output)
            for name, artifact in artifacts.items()
        },
        "policies": sorted(
            {
                policy
                for artifact in artifacts.values()
                for policy in artifact.raw_results["policy"].unique()
            }
        ),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_consistency": dict(baseline_consistency),
        **reproducibility_metadata(),
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    report = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Phase 3 Robustness and Sensitivity Analysis — Mode 4</title>
<style>{REPORT_STYLE}
h3 {{ color: #334e75; }}
.plots {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 1rem; }}
figure {{ margin: 0; border: 1px solid #c9d0d9; padding: .6rem; background: #fff; }}
figure img {{ width: 100%; height: auto; }}
figcaption {{ font-size: .85rem; color: #4b5563; text-align: center; }}
.table-scroll {{ max-width: 100%; overflow-x: auto; }}
@media (max-width: 520px) {{ .plots {{ grid-template-columns: 1fr; }} }}
</style></head><body>
<h1>Phase 3 Robustness and Sensitivity Analysis — Mode 4</h1>
<p class="note">This report contains only the existing Phase 3 CMDP policy family. Phase 1 arm policies and Phase 2 contextual-bandit policies were not run. Opportunity counts remain the denominators for demographic parity. Feasibility screens are reported separately from whether a policy satisfied a feasible constraint.</p>
<p><strong>Run type:</strong> {'quick smoke test' if quick_test else 'full experiment'}; <strong>seeds:</strong> {escape(', '.join(map(str, seeds)))}.</p>
<p class="winner">Baseline consistency check: {'passed' if baseline_consistency.get('all_passed') else 'failed'}.
This check compares the budget default (multiplier 1.0) with the group-imbalance default (marginalized probability 0.20)
and the fairness-strength default (strength 1.0) for matching policy and seed rows.
Comparison with a standalone Mode 3 run is planned for Phase 4.</p>
{''.join(sections)}
<h2>Reproducibility manifest</h2><pre>{escape(json.dumps(manifest, indent=2, sort_keys=True, default=str))}</pre>
</body></html>"""
    path = output / "report.html"
    path.write_text(report, encoding="utf-8")
    return path
