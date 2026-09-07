from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from html import escape
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import platform

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Algorithms.bandit import BasePolicy
from Algorithms.cmdp import CMDPPolicy, LagrangianFairCMDPPolicy
from Environments.base_environment import BanditEnvironment
from Environments.cmdp_environment import StatefulMicrofinanceCMDPEnv
from metrics import (
    allocation_shares,
    budget_utilization,
    capital_survival_ratio,
    capital_turnover,
    demographic_parity_gap,
    group_funding_rates,
    liquidity_utilization,
    quota_deficits,
    quota_satisfaction_rate,
    repayment_rate,
    reward_per_disbursed_capital,
    reward_per_unit_cost,
    total_quota_deficit,
)
from models.environment_config import CMDPEnvironmentConfig


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    """Represents the result of running a single episode, including a summary of metrics and the trajectory of actions taken."""
    summary: dict[str, float | int | str]
    trajectory: pd.DataFrame


def run_episode(
    env: BanditEnvironment,
    policy: BasePolicy,
    target_shares: Mapping[str, float] | None = None,
    seed: int | None = None,
) -> EpisodeResult:
    """Run a single episode of the environment with the given policy and return the summary and trajectory."""
    targets = dict(target_shares or {})
    env.reset(seed)
    policy.seed = env.config.seed if seed is None else seed
    policy.reset()
    rows: list[dict[str, float | int | str]] = []
    cumulative_reward = 0.0
    cumulative_cost = 0.0

    while not env.done:
        action = policy.select_action(env)
        result = env.step(action)
        for feedback in result.observed_feedback:
            policy.update(feedback)
        cumulative_reward += result.reward
        cumulative_cost += result.cost
        shares = allocation_shares(result.funding_counts)
        rates = group_funding_rates(
            result.funding_counts, result.opportunity_counts
        )
        deficits = quota_deficits(result.funding_counts, targets)
        decisions = result.round_index + 1
        row: dict[str, float | int | str] = {
            "round": result.round_index,
            "selected_index": result.selected_index,
            "identifier": result.identifier,
            "fairness_group": result.group,
            "reward": result.reward,
            "cost": result.cost,
            "cumulative_reward": cumulative_reward,
            "cumulative_cost": cumulative_cost,
            "budget_remaining": result.budget_remaining,
            "budget_utilization": budget_utilization(
                env.config.total_budget, result.budget_remaining
            ),
            "reward_per_cost": (
                cumulative_reward / cumulative_cost if cumulative_cost else 0.0
            ),
            "repayment_rate": cumulative_reward / decisions,
            "demographic_parity_gap": demographic_parity_gap(
                result.funding_counts, result.opportunity_counts
            ),
            "quota_satisfaction_rate": quota_satisfaction_rate(
                result.funding_counts, targets
            ),
            "total_quota_deficit": float(sum(deficits.values())),
            "observed_feedback_count": len(result.observed_feedback),
        }
        if result.true_repayment_probability is not None:
            row["true_repayment_probability"] = result.true_repayment_probability
        for key, value in result.metadata.items():
            if isinstance(value, (str, int, float)):
                row[key] = value
        for group in sorted(set(result.opportunity_counts) | set(targets)):
            row[f"funded_{group}"] = result.funding_counts.get(group, 0)
            row[f"opportunities_{group}"] = result.opportunity_counts.get(group, 0)
            row[f"allocation_share_{group}"] = shares.get(group, 0.0)
            row[f"funding_rate_{group}"] = rates.get(group, 0.0)
            row[f"quota_deficit_{group}"] = deficits.get(group, 0.0)
        rows.append(row)

    trajectory = pd.DataFrame(rows)
    rewards = (
        trajectory["reward"].to_numpy() if not trajectory.empty else np.array([])
    )
    costs = trajectory["cost"].to_numpy() if not trajectory.empty else np.array([])
    final_shares = allocation_shares(env.group_funding_counts)
    final_rates = group_funding_rates(
        env.group_funding_counts, env.group_opportunity_counts
    )
    summary: dict[str, float | int | str] = {
        "rounds": env.total_allocations,
        "cumulative_reward": float(np.sum(rewards)),
        "cumulative_cost": float(np.sum(costs)),
        "reward_per_cost": reward_per_unit_cost(rewards, costs),
        "repayment_rate": repayment_rate(rewards),
        "budget_utilization": budget_utilization(
            env.config.total_budget, env.remaining_budget
        ),
        "demographic_parity_gap": demographic_parity_gap(
            env.group_funding_counts, env.group_opportunity_counts
        ),
        "quota_satisfaction_rate": quota_satisfaction_rate(
            env.group_funding_counts, targets
        ),
        "total_quota_deficit": total_quota_deficit(
            env.group_funding_counts, targets
        ),
        "pending_feedback": len(getattr(env, "pending_feedback", [])),
    }
    for group in sorted(env.group_opportunity_counts):
        summary[f"funded_{group}"] = env.group_funding_counts.get(group, 0)
        summary[f"opportunities_{group}"] = env.group_opportunity_counts.get(group, 0)
        summary[f"allocation_share_{group}"] = final_shares.get(group, 0.0)
        summary[f"funding_rate_{group}"] = final_rates.get(group, 0.0)
    return EpisodeResult(summary=summary, trajectory=trajectory)


EnvironmentFactory = Callable[[int], BanditEnvironment]
PolicyFactory = Callable[[BanditEnvironment, int], BasePolicy]


def compare_policies(
    environment_factory: EnvironmentFactory,
    policy_factories: Mapping[str, PolicyFactory],
    target_shares: Mapping[str, float],
    seeds: Sequence[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run episodes for each policy and seed, collecting summaries and trajectories."""
    summaries: list[dict[str, float | int | str]] = []
    trajectories: list[pd.DataFrame] = []
    for policy_name, factory in policy_factories.items():
        for seed in seeds:
            env = environment_factory(seed)
            policy = factory(env, seed)
            episode = run_episode(env, policy, target_shares, seed)
            summary = dict(episode.summary)
            summary.update({"policy": policy_name, "seed": seed})
            summaries.append(summary)
            trajectory = episode.trajectory.copy()
            trajectory["policy"] = policy_name
            trajectory["seed"] = seed
            trajectories.append(trajectory)
    return pd.DataFrame(summaries), pd.concat(trajectories, ignore_index=True)


# need to check, these metrics and whether they are reliable or need to add more metrics to evaluate the performance of the policies. The current metrics are:
COMMON_METRICS = (
    "cumulative_reward",
    "cumulative_cost",
    "reward_per_cost",
    "repayment_rate",
    "budget_utilization",
    "demographic_parity_gap",
    "quota_satisfaction_rate",
    "total_quota_deficit",
)


def aggregate_results(summaries: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the results by policy, computing the mean and standard deviation for each common metric."""
    metrics = [metric for metric in COMMON_METRICS if metric in summaries]
    aggregate = summaries.groupby("policy")[metrics].agg(["mean", "std"])
    std_columns = [column for column in aggregate.columns if column[1] == "std"]
    aggregate.loc[:, std_columns] = aggregate.loc[:, std_columns].fillna(0.0)
    return aggregate.sort_values(("cumulative_reward", "mean"), ascending=False)


def plot_metric_over_time(
    trajectories: pd.DataFrame,
    metric: str,
    output_path: Path,
    mode: str,
) -> None:
    """Plot the specified metric over time for each policy and save the plot to the given output path."""
    averaged = trajectories.groupby(["policy", "round"], as_index=False)[metric].mean()
    fig, ax = plt.subplots(figsize=(10, 6))
    for policy, frame in averaged.groupby("policy"):
        ax.plot(frame["round"], frame[metric], label=policy)
    label = metric.replace("_", " ").title()
    ax.set_xlabel("Round")
    ax.set_ylabel(label)
    family = (
        f"{mode.title()} bandits"
        if mode in {"arm", "contextual"}
        else mode.upper()
    )
    ax.set_title(f"{family}: {label} over time")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_results(
    summaries: pd.DataFrame,
    trajectories: pd.DataFrame,
    output: Path,
    mode: str,
) -> pd.DataFrame:
    """Save the episode summaries, trajectories, and aggregate results to CSV files and generate plots for common metrics."""
    output.mkdir(parents=True, exist_ok=True)
    summaries.to_csv(output / "episode_summaries.csv", index=False)
    trajectories.to_csv(output / "trajectories.csv", index=False)
    aggregate = aggregate_results(summaries)
    aggregate.to_csv(output / "aggregate_results.csv")
    for metric in (
        "cumulative_reward",
        "repayment_rate",
        "demographic_parity_gap",
        "quota_satisfaction_rate",
    ):
        if metric in trajectories:
            plot_metric_over_time(
                trajectories, metric, output / f"{metric}.png", mode
            )
    return aggregate


def display_results(
    aggregate: pd.DataFrame,
    output: Path,
    mode: str,
    ranked: pd.DataFrame | None = None,
) -> None:
    """Display aggregate or ranked results and indicate where outputs were saved."""

    label = f"{mode.title()} bandit" if mode in {"arm", "contextual"} else mode.upper()
    print(f"\n{label} aggregate results:\n")
    if ranked is None:
        print(aggregate.round(5).to_string())
    else:
        print(format_ranked_table(ranked, mode))
        print(f"\n{winner_summary(ranked, mode)}")
    print(f"\nSaved outputs to: {output.resolve()}")


@dataclass(frozen=True, slots=True)
class CMDPEpisodeResult:
    """Phase 3 result, including portfolio-level audit logs."""

    summary: dict[str, float | int | str]
    trajectory: pd.DataFrame
    funding_log: pd.DataFrame
    maturity_log: pd.DataFrame


def _policy_lambdas(policy: CMDPPolicy) -> tuple[float, float]:
    if isinstance(policy, LagrangianFairCMDPPolicy):
        return policy.lambda_quota, policy.lambda_parity
    return 0.0, 0.0


def run_cmdp_episode(
    env: StatefulMicrofinanceCMDPEnv,
    policy: CMDPPolicy,
    target_shares: Mapping[str, float],
    parity_tolerance: float,
    seed: int,
) -> CMDPEpisodeResult:
    """Run one stateful Phase 3 lending episode."""

    env.reset(seed)
    policy.seed = seed
    policy.reset()
    rows: list[dict[str, float | int | str]] = []

    while not env.done:
        matured_now = env.matured_at_round_start
        for feedback in matured_now:
            policy.update(
                feedback.learning_context,
                feedback.reward,
                feedback.principal,
            )

        funded_this_round = 0
        hidden_reward_this_round = 0.0
        disbursed_this_round = 0.0
        while funded_this_round < env.config.max_loans_per_round:
            action = policy.select_action(env)
            if action is None:
                break
            learning_context = policy.decision_context(env, action)
            result = env.fund(action, learning_context)
            funded_this_round += 1
            hidden_reward_this_round += result.hidden_reward
            disbursed_this_round += result.principal

        fairness = env.fairness_snapshot(target_shares)
        lambda_quota, lambda_parity = _policy_lambdas(policy)
        rows.append(
            {
                "round": env.round_index,
                "funded_this_round": funded_this_round,
                "matured_this_round": len(matured_now),
                "hidden_reward_funded_this_round": hidden_reward_this_round,
                "disbursed_this_round": disbursed_this_round,
                "cumulative_funded": env.total_allocations,
                "eventual_cumulative_reward": env.eventual_reward(),
                "observed_repayments": env.observed_repayments,
                "observed_defaults": env.observed_defaults,
                "observed_repayment_rate": env.observed_repayment_rate,
                "recent_repayment_rate": env.recent_repayment_rate,
                "available_capital": env.available_capital,
                "capital_at_risk": env.capital_at_risk,
                "pending_loans": len(env.pending_loans),
                "cumulative_disbursed": env.cumulative_disbursed,
                "cumulative_recycled": env.cumulative_recycled,
                "cumulative_default_loss": env.cumulative_default_loss,
                "capital_turnover": capital_turnover(
                    env.cumulative_disbursed,
                    env.config.initial_capital,
                ),
                "liquidity_utilization": liquidity_utilization(
                    env.available_capital,
                    env.capital_at_risk,
                ),
                "capital_survival_ratio": capital_survival_ratio(
                    env.available_capital,
                    env.capital_at_risk,
                    env.config.initial_capital,
                ),
                "demographic_parity_gap": fairness[
                    "demographic_parity_gap"
                ],
                "total_quota_deficit": fairness["total_quota_deficit"],
                "quota_satisfaction_rate": quota_satisfaction_rate(
                    env.group_funding_counts,
                    target_shares,
                ),
                "lambda_quota": lambda_quota,
                "lambda_parity": lambda_parity,
                "state_bucket": env.state_bucket(
                    target_shares,
                    parity_tolerance,
                ),
            }
        )
        policy.on_round_end(env)
        if env.advance_round().done:
            break

    eventual_rewards = np.asarray(
        [float(row["hidden_reward"]) for row in env.funding_history],
        dtype=float,
    )
    principals = np.asarray(
        [float(row["principal"]) for row in env.funding_history],
        dtype=float,
    )
    eventual_terminal_capital = (
        env.available_capital + env.eventual_pending_capital_return()
    )
    final_rates = group_funding_rates(
        env.group_funding_counts,
        env.group_opportunity_counts,
    )
    final_shares = allocation_shares(env.group_funding_counts)
    final_deficits = quota_deficits(
        env.group_funding_counts,
        target_shares,
    )
    lambda_quota, lambda_parity = _policy_lambdas(policy)

    summary: dict[str, float | int | str] = {
        "rounds": env.round_index,
        "loans_funded": env.total_allocations,
        "cumulative_reward": float(np.sum(eventual_rewards)),
        "observed_reward_by_horizon": float(env.observed_repayments),
        "repayment_rate": repayment_rate(eventual_rewards),
        "cumulative_disbursed": env.cumulative_disbursed,
        "reward_per_cost": reward_per_disbursed_capital(
            eventual_rewards,
            principals,
        ),
        "capital_turnover": capital_turnover(
            env.cumulative_disbursed,
            env.config.initial_capital,
        ),
        "cumulative_recycled": env.cumulative_recycled,
        "default_loss_by_horizon": env.cumulative_default_loss,
        "final_available_capital": env.available_capital,
        "final_capital_at_risk": env.capital_at_risk,
        "eventual_terminal_capital": eventual_terminal_capital,
        "capital_survival_ratio": capital_survival_ratio(
            env.available_capital,
            env.capital_at_risk,
            env.config.initial_capital,
        ),
        "demographic_parity_gap": demographic_parity_gap(
            env.group_funding_counts,
            env.group_opportunity_counts,
        ),
        "quota_satisfaction_rate": quota_satisfaction_rate(
            env.group_funding_counts,
            target_shares,
        ),
        "total_quota_deficit": total_quota_deficit(
            env.group_funding_counts,
            target_shares,
        ),
        "pending_loans": len(env.pending_loans),
        "lambda_quota_final": lambda_quota,
        "lambda_parity_final": lambda_parity,
    }
    for group in sorted(env.group_opportunity_counts):
        summary[f"funded_{group}"] = env.group_funding_counts.get(group, 0)
        summary[f"opportunities_{group}"] = env.group_opportunity_counts.get(
            group, 0
        )
        summary[f"allocation_share_{group}"] = final_shares.get(group, 0.0)
        summary[f"funding_rate_{group}"] = final_rates.get(group, 0.0)
        summary[f"quota_deficit_{group}"] = final_deficits.get(group, 0.0)

    return CMDPEpisodeResult(
        summary=summary,
        trajectory=pd.DataFrame(rows),
        funding_log=pd.DataFrame(env.funding_history),
        maturity_log=pd.DataFrame(env.maturity_history),
    )


CMDPPolicyFactory = Callable[[int, int], CMDPPolicy]


def compare_cmdp_policies(
    env_config: CMDPEnvironmentConfig,
    policy_factories: Mapping[str, CMDPPolicyFactory],
    target_shares: Mapping[str, float],
    parity_tolerance: float,
    seeds: Sequence[int],
    encoder: "BorrowerFeatureEncoder | None" = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate all Phase 3 policies using the shared borrower representation."""

    # Imported locally to keep the Phase 1/2 evaluation path lightweight.
    from Data.feature_encoder import BorrowerFeatureEncoder

    shared_encoder = encoder or BorrowerFeatureEncoder(
        include_protected_attributes=False
    )
    summaries: list[dict[str, float | int | str]] = []
    trajectories: list[pd.DataFrame] = []
    funding_logs: list[pd.DataFrame] = []
    maturity_logs: list[pd.DataFrame] = []

    for policy_name, factory in policy_factories.items():
        for seed in seeds:
            env = StatefulMicrofinanceCMDPEnv(env_config, shared_encoder)
            policy = factory(shared_encoder.n_features, seed)
            episode = run_cmdp_episode(
                env,
                policy,
                target_shares,
                parity_tolerance,
                seed,
            )
            summary = dict(episode.summary)
            summary.update({"policy": policy_name, "seed": seed})
            summaries.append(summary)

            for frame, collection in (
                (episode.trajectory, trajectories),
                (episode.funding_log, funding_logs),
                (episode.maturity_log, maturity_logs),
            ):
                copied = frame.copy()
                copied["policy"] = policy_name
                copied["seed"] = seed
                collection.append(copied)

    def concat_or_empty(frames: list[pd.DataFrame]) -> pd.DataFrame:
        return (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame()
        )

    return (
        pd.DataFrame(summaries),
        concat_or_empty(trajectories),
        concat_or_empty(funding_logs),
        concat_or_empty(maturity_logs),
    )


CMDP_METRICS = (
    "loans_funded",
    "cumulative_reward",
    "observed_reward_by_horizon",
    "repayment_rate",
    "reward_per_cost",
    "capital_turnover",
    "cumulative_recycled",
    "default_loss_by_horizon",
    "final_available_capital",
    "final_capital_at_risk",
    "eventual_terminal_capital",
    "capital_survival_ratio",
    "demographic_parity_gap",
    "quota_satisfaction_rate",
    "total_quota_deficit",
    "lambda_quota_final",
    "lambda_parity_final",
)


def aggregate_cmdp_results(summaries: pd.DataFrame) -> pd.DataFrame:
    metrics = [metric for metric in CMDP_METRICS if metric in summaries]
    aggregate = summaries.groupby("policy")[metrics].agg(["mean", "std"])
    std_columns = [column for column in aggregate.columns if column[1] == "std"]
    aggregate.loc[:, std_columns] = aggregate.loc[:, std_columns].fillna(0.0)
    return aggregate.sort_values(
        ("cumulative_reward", "mean"),
        ascending=False,
    )


def save_cmdp_results(
    summaries: pd.DataFrame,
    trajectories: pd.DataFrame,
    funding_log: pd.DataFrame,
    maturity_log: pd.DataFrame,
    output: Path,
) -> pd.DataFrame:
    """Save Phase 3 metrics, trajectories, and portfolio audit logs."""

    output.mkdir(parents=True, exist_ok=True)
    summaries.to_csv(output / "episode_summaries.csv", index=False)
    trajectories.to_csv(output / "trajectories.csv", index=False)
    funding_log.to_csv(output / "funding_log.csv", index=False)
    maturity_log.to_csv(output / "maturity_log.csv", index=False)
    aggregate = aggregate_cmdp_results(summaries)
    aggregate.to_csv(output / "aggregate_results.csv")
    for metric in (
        "eventual_cumulative_reward",
        "available_capital",
        "capital_at_risk",
        "capital_turnover",
        "demographic_parity_gap",
        "quota_satisfaction_rate",
        "lambda_quota",
        "lambda_parity",
    ):
        if metric in trajectories:
            plot_metric_over_time(
                trajectories,
                metric,
                output / f"{metric}.png",
                "cmdp",
            )
    return aggregate


def _flatten_aggregate(aggregate: pd.DataFrame) -> pd.DataFrame:
    flattened = aggregate.copy()
    if isinstance(flattened.columns, pd.MultiIndex):
        flattened.columns = [
            "_".join(str(part) for part in column if part)
            for column in flattened.columns
        ]
    flattened = flattened.reset_index()
    if "policy" not in flattened and len(flattened.columns):
        flattened = flattened.rename(columns={flattened.columns[0]: "policy"})
    return flattened


def rank_results(
    aggregate: pd.DataFrame,
    parity_tolerance: float,
    quota_tolerance: float = 0.01,
) -> pd.DataFrame:
    """Rank policies by fairness feasibility, reward efficiency, and reward."""

    if not 0.0 <= parity_tolerance <= 1.0:
        raise ValueError("parity_tolerance must be in [0, 1]")
    if quota_tolerance < 0.0:
        raise ValueError("quota_tolerance cannot be negative")
    ranked = _flatten_aggregate(aggregate)
    required = {
        "policy",
        "cumulative_reward_mean",
        "reward_per_cost_mean",
        "demographic_parity_gap_mean",
        "total_quota_deficit_mean",
    }
    missing = sorted(required - set(ranked.columns))
    if missing:
        raise ValueError(f"aggregate results are missing ranking metrics: {missing}")

    ranked["parity_excess"] = np.maximum(
        0.0,
        ranked["demographic_parity_gap_mean"] - parity_tolerance,
    )
    ranked["quota_excess"] = np.maximum(
        0.0,
        ranked["total_quota_deficit_mean"] - quota_tolerance,
    )
    ranked["fairness_violation"] = (
        ranked["parity_excess"] + ranked["quota_excess"]
    )
    ranked["fairness_feasible"] = ranked["fairness_violation"] <= 1e-12
    ranked["_feasibility_group"] = np.where(
        ranked["fairness_feasible"], 0, 1
    )
    ranked["_violation_order"] = np.where(
        ranked["fairness_feasible"], 0.0, ranked["fairness_violation"]
    )
    ranked = ranked.sort_values(
        [
            "_feasibility_group",
            "_violation_order",
            "reward_per_cost_mean",
            "cumulative_reward_mean",
            "policy",
        ],
        ascending=[True, True, False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1, dtype=int))
    return ranked.drop(columns=["_feasibility_group", "_violation_order"])


def winner_summary(ranked: pd.DataFrame, mode: str) -> str:
    """Describe the deterministic winner selected from a ranked result table."""

    if ranked.empty:
        return f"{mode.title()} has no policies to rank."
    winner = ranked.iloc[0]
    label = "Phase 2 contextual" if mode == "contextual" else "Phase 3 CMDP"
    reward_per_cost = float(winner["reward_per_cost_mean"])
    cumulative_reward = float(winner["cumulative_reward_mean"])
    parity_gap = float(winner["demographic_parity_gap_mean"])
    quota_deficit = float(winner["total_quota_deficit_mean"])
    if bool(winner["fairness_feasible"]):
        qualification = "It satisfies the configured parity and quota thresholds"
    else:
        qualification = (
            "It is the best compromise; no fairness-feasible policy was found"
        )
    return (
        f"{label} winner: {winner['policy']}. {qualification}, with mean "
        f"reward-per-cost {reward_per_cost:.6f}, cumulative reward "
        f"{cumulative_reward:.3f}, parity gap {parity_gap:.6f}, and total "
        f"quota deficit {quota_deficit:.6f}."
    )


def _format_table_value(value: object) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "yes" if bool(value) else "no"
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return "-"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.5f}"
    return str(value)


def _bordered_table(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    values = [
        [_format_table_value(value) for value in row]
        for row in frame.itertuples(index=False, name=None)
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in values))
        for index in range(len(headers))
    ]
    border = "+" + "+".join("-" * (width + 2) for width in widths) + "+"

    def render(row: Sequence[str]) -> str:
        return "| " + " | ".join(
            value.ljust(width) for value, width in zip(row, widths)
        ) + " |"

    lines = [border, render(headers), border]
    for row in values:
        lines.extend((render(row), border))
    return "\n".join(lines)


def format_ranked_table(ranked: pd.DataFrame, mode: str) -> str:
    """Build a compact dependency-free ASCII grid for terminal display."""

    columns = {
        "rank": "Rank",
        "policy": "Policy",
        "fairness_feasible": "Fair",
        "cumulative_reward_mean": "Reward",
        "reward_per_cost_mean": "Reward/Cost",
        "repayment_rate_mean": "Repayment",
        "demographic_parity_gap_mean": "Parity Gap",
        "total_quota_deficit_mean": "Quota Deficit",
    }
    if mode == "contextual":
        columns["budget_utilization_mean"] = "Budget Use"
    if mode == "cmdp":
        columns["capital_turnover_mean"] = "Turnover"
        columns["capital_survival_ratio_mean"] = "Capital Survival"
    available = {key: value for key, value in columns.items() if key in ranked}
    return _bordered_table(ranked[list(available)].rename(columns=available))


def _package_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for package in ("numpy", "pandas", "matplotlib"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def _mean_std_column(ranked: pd.DataFrame, metric: str) -> pd.Series:
    means = ranked[f"{metric}_mean"]
    standard_deviations = ranked.get(
        f"{metric}_std", pd.Series(0.0, index=ranked.index)
    )
    return pd.Series(
        [
            f"{float(mean):.6f} ± {float(std):.6f}"
            for mean, std in zip(means, standard_deviations)
        ],
        index=ranked.index,
    )


def _html_ranked_table(ranked: pd.DataFrame, mode: str) -> str:
    display = pd.DataFrame(
        {
            "Rank": ranked["rank"],
            "Policy": ranked["policy"],
            "Fairness feasible": ranked["fairness_feasible"].map(
                {True: "Yes", False: "No"}
            ),
            "Reward (mean ± std)": _mean_std_column(
                ranked, "cumulative_reward"
            ),
            "Reward/cost (mean ± std)": _mean_std_column(
                ranked, "reward_per_cost"
            ),
            "Repayment (mean ± std)": _mean_std_column(
                ranked, "repayment_rate"
            ),
            "Parity gap (mean ± std)": _mean_std_column(
                ranked, "demographic_parity_gap"
            ),
            "Quota deficit (mean ± std)": _mean_std_column(
                ranked, "total_quota_deficit"
            ),
        }
    )
    if mode == "contextual" and "budget_utilization_mean" in ranked:
        display["Budget use (mean ± std)"] = _mean_std_column(
            ranked, "budget_utilization"
        )
    if mode == "cmdp":
        display["Capital turnover (mean ± std)"] = _mean_std_column(
            ranked, "capital_turnover"
        )
        display["Capital survival (mean ± std)"] = _mean_std_column(
            ranked, "capital_survival_ratio"
        )
    return display.to_html(index=False, border=0, classes="results", escape=True)


REPORT_STYLE = """
body { font-family: system-ui, sans-serif; margin: 2rem; color: #172033; }
h1, h2 { color: #203864; }
.winner { padding: 1rem; background: #eaf4e8; border-left: 5px solid #3d7d35; }
.note { padding: .8rem; background: #f2f5f9; border-left: 5px solid #68778d; }
table.results { border-collapse: collapse; width: 100%; font-size: .88rem; }
table.results th, table.results td { border: 1px solid #6b7280; padding: .45rem; text-align: right; }
table.results th:nth-child(2), table.results td:nth-child(2) { text-align: left; }
table.results th { background: #203864; color: white; position: sticky; top: 0; }
table.results tr:nth-child(even) { background: #f5f7fa; }
pre { background: #f5f7fa; border: 1px solid #c9d0d9; padding: 1rem; overflow: auto; }
a { color: #174ea6; }
""".strip()


def save_evaluation_report(
    aggregate: pd.DataFrame,
    output: Path,
    mode: str,
    parity_tolerance: float,
    seeds: Sequence[int],
    configuration: Mapping[str, object],
    policy_parameters: Mapping[str, object],
    quota_tolerance: float = 0.01,
) -> pd.DataFrame:
    """Rank results and persist reproducibility and human-readable reports."""

    output.mkdir(parents=True, exist_ok=True)
    ranked = rank_results(aggregate, parity_tolerance, quota_tolerance)
    summary = winner_summary(ranked, mode)
    ranked.to_csv(output / "ranked_results.csv", index=False)
    (output / "winner_summary.txt").write_text(summary + "\n", encoding="utf-8")
    manifest = {
        "mode": mode,
        "seeds": list(seeds),
        "policies": ranked["policy"].tolist(),
        "fairness_thresholds": {
            "parity_tolerance": parity_tolerance,
            "quota_tolerance": quota_tolerance,
        },
        "configuration": dict(configuration),
        "policy_parameters": dict(policy_parameters),
        "package_versions": _package_versions(),
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    title = "Phase 2 Contextual Bandits" if mode == "contextual" else "Phase 3 CMDP"
    report = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(title)} evaluation</title><style>{REPORT_STYLE}</style></head>
<body><h1>{escape(title)} evaluation</h1>
<p class="winner">{escape(summary)}</p>
<p class="note">Policies are fairness-feasible when mean demographic parity gap is at most {parity_tolerance:.6f} and mean total quota deficit is at most {quota_tolerance:.6f}. Feasible policies rank by reward-per-cost, then cumulative reward. Infeasible policies rank by constraint violation before utility.</p>
<h2>Ranked results</h2>{_html_ranked_table(ranked, mode)}
<h2>Reproducibility manifest</h2><pre>{escape(json.dumps(manifest, indent=2, sort_keys=True, default=str))}</pre>
</body></html>"""
    (output / "report.html").write_text(report, encoding="utf-8")
    return ranked


def save_combined_report(
    ranked_results: Mapping[str, pd.DataFrame],
    output: Path,
    arm_aggregate: pd.DataFrame | None = None,
    sweep_report: Path | None = None,
) -> Path:
    """Save the Mode 5 suite summary without comparing incompatible modes."""

    output.mkdir(parents=True, exist_ok=True)
    sections: list[str] = []
    if arm_aggregate is not None and not arm_aggregate.empty:
        arm = _flatten_aggregate(arm_aggregate)
        arm = arm.sort_values("cumulative_reward_mean", ascending=False)
        columns = [
            column
            for column in (
                "policy",
                "cumulative_reward_mean",
                "reward_per_cost_mean",
                "repayment_rate_mean",
                "demographic_parity_gap_mean",
                "quota_satisfaction_rate_mean",
            )
            if column in arm
        ]
        arm_table = arm[columns].rename(
            columns={
                "policy": "Policy",
                "cumulative_reward_mean": "Mean reward",
                "reward_per_cost_mean": "Mean reward/cost",
                "repayment_rate_mean": "Mean repayment",
                "demographic_parity_gap_mean": "Mean parity gap",
                "quota_satisfaction_rate_mean": "Mean quota satisfaction",
            }
        )
        sections.append(
            "<h2>Mode 1 — Arm Bandits</h2>"
            '<p class="note">Policies are ordered by mean cumulative reward; '
            "this is a within-mode description, not a cross-mode ranking.</p>"
            f"{arm_table.to_html(index=False, border=0, classes='results arm-results', escape=True)}"
            '<p><a href="arm/aggregate_results.csv">Open Mode 1 aggregate results</a></p>'
        )
    for mode in ("contextual", "cmdp"):
        ranked = ranked_results.get(mode)
        if ranked is None or ranked.empty:
            continue
        label = (
            "Mode 2 — Phase 2 Contextual"
            if mode == "contextual"
            else "Mode 3 — Phase 3 CMDP"
        )
        sections.append(
            f"<h2>{escape(label)}</h2><p class=\"winner\">"
            f"{escape(winner_summary(ranked, mode))}</p>"
            f"<p><a href=\"{mode}/report.html\">Open the full {escape(label)} report</a></p>"
        )
    if sweep_report is not None:
        try:
            sweep_href = sweep_report.relative_to(output).as_posix()
        except ValueError:
            sweep_href = sweep_report.as_posix()
        sections.append(
            "<h2>Mode 4 — Phase 3 CMDP Robustness Sweeps</h2>"
            '<p class="note">Budget, group-imbalance, and fairness-strength '
            "sweeps use only the existing Phase 3 CMDP policy family.</p>"
            f'<p><a href="{escape(sweep_href)}">Open the full Mode 4 robustness report</a></p>'
        )
    report = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mode 5 — Complete experiment suite</title><style>{REPORT_STYLE}
table.arm-results th:first-child, table.arm-results td:first-child {{ text-align: left; }}
</style></head>
<body><h1>Mode 5 — Complete experiment suite</h1>
<p class="note">Modes 1–4 have different policy families, budget semantics, and capital lifecycles. Results are collected in one directory, but each mode is interpreted separately and no cross-mode winner is declared.</p>
<p><a href="combined_aggregate_results.csv">Open the Modes 1–3 combined aggregate table</a> · <a href="run_manifest.json">Open the Mode 5 reproducibility manifest</a></p>
{''.join(sections)}</body></html>"""
    path = output / "report.html"
    path.write_text(report, encoding="utf-8")
    return path
