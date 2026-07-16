from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Algorithms.bandit import BasePolicy
from Environments.base_environment import BanditEnvironment
from metrics import (
    allocation_shares,
    budget_utilization,
    demographic_parity_gap,
    group_funding_rates,
    quota_deficits,
    quota_satisfaction_rate,
    repayment_rate,
    reward_per_unit_cost,
)


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
    ax.set_title(f"{mode.title()} bandits: {label} over time")
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
    print(f"\n{mode.title()} bandit aggregate results:\n")
    print(aggregate.round(5).to_string())
    print(f"\nSaved outputs to: {output.resolve()}")
