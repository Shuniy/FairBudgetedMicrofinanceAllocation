from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

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


def display_results(aggregate: pd.DataFrame, output: Path, mode: str) -> None:
    """Display the aggregate results in a readable format and indicate where the outputs were saved."""
    label = f"{mode.title()} bandit" if mode in {"arm", "contextual"} else mode.upper()
    print(f"\n{label} aggregate results:\n")
    print(aggregate.round(5).to_string())
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
