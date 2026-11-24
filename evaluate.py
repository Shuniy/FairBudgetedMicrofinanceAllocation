from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from html import escape
from importlib.metadata import PackageNotFoundError, version
from inspect import signature
from pathlib import Path
import platform
import subprocess
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Algorithms.bandit import BasePolicy
from Algorithms.cmdp import CMDPPolicy, LagrangianFairCMDPPolicy, OraclePortfolioPolicy
from Algorithms.contextual import TrueRewardContextualPolicy
from Algorithms.true_repayment_cost import TrueRewardPerCostPolicy
from Environments.base_environment import BanditEnvironment, BanditMode
from Environments.cmdp_environment import StatefulMicrofinanceCMDPEnv
from episode_statistics import CI_METHODS, describe, paired_comparisons
from metrics import (
    allocation_share_gap,
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
)
from models.environment_config import CMDPEnvironmentConfig


ARM_PARITY_TOLERANCE = 0.35
CONTEXTUAL_PARITY_TOLERANCE = 0.010
QUOTA_TOLERANCE = 0.01


def episode_constraint_outcomes(
    parity_gap: float,
    total_deficit: float,
    parity_tolerance: float,
    quota_tolerance: float,
) -> dict[str, float | bool]:
    """Evaluate one episode or trajectory snapshot against its thresholds."""
    evaluable = not np.isnan(parity_gap)
    quota_satisfied = total_deficit <= quota_tolerance
    parity_satisfied = parity_gap <= parity_tolerance
    return {
        "parity_evaluable": evaluable,
        "parity_satisfied": int(parity_satisfied) if evaluable else np.nan,
        "quota_satisfied": quota_satisfied,
        "constraint_satisfied": int(parity_satisfied and quota_satisfied) if evaluable else np.nan,
    }


def _fairness_metrics(
    funded: Mapping[str, int],
    opportunities: Mapping[str, int],
    population: Mapping[str, int] | None,
    targets: Mapping[str, float],
    parity_tolerance: float,
    quota_tolerance: float,
) -> dict[str, float | int]:
    """Build the fairness columns shared by trajectory rows and episode summaries."""
    shares = allocation_shares(funded)
    rates = group_funding_rates(funded, opportunities)
    deficits = quota_deficits(funded, targets)
    total_deficit = float(sum(deficits.values()))
    fields = {
        "quota_satisfaction_rate": quota_satisfaction_rate(funded, targets),
        "total_quota_deficit": total_deficit,
    }
    groups = set(funded) | set(opportunities) | set(targets)
    if population is None:
        gap = allocation_share_gap(funded)
        fields["allocation_share_gap"] = gap
        constraint_gap = gap if sum(funded.values()) else np.nan
    else:
        gap = demographic_parity_gap(funded, opportunities)
        constraint_gap = gap
        population_rates = group_funding_rates(funded, population)
        population_gap = demographic_parity_gap(funded, population)
        population_outcomes = episode_constraint_outcomes(
            population_gap, total_deficit, parity_tolerance, quota_tolerance
        )
        fields.update({
            "feasible_dp_gap": gap,
            "feasible_dp_groups": len(rates),
            "population_dp_gap": population_gap,
            "population_dp_groups": len(population_rates),
            **{
                f"population_{name}": value
                for name, value in population_outcomes.items()
                if name != "quota_satisfied"
            },
        })
        groups.update(population)
        for group in sorted(groups):
            fields[f"population_opportunities_{group}"] = population.get(group, 0)
            fields[f"population_funding_rate_{group}"] = population_rates.get(group, np.nan)

    fields["demographic_parity_gap"] = gap
    fields.update(episode_constraint_outcomes(
        constraint_gap, total_deficit, parity_tolerance, quota_tolerance
    ))
    for group in sorted(groups):
        fields[f"funded_{group}"] = funded.get(group, 0)
        fields[f"opportunities_{group}"] = opportunities.get(group, 0)
        fields[f"allocation_share_{group}"] = shares.get(group, 0.0)
        fields[f"funding_rate_{group}"] = rates.get(group, 0.0 if population is None else np.nan)
        fields[f"quota_deficit_{group}"] = deficits.get(group, 0.0)
    return fields


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    """Represents the result of running a single episode, including a summary of metrics and the trajectory of actions taken."""
    summary: dict[str, float | int | str]
    trajectory: pd.DataFrame


def _uses_true_probabilities(policy: BasePolicy | CMDPPolicy) -> bool:
    while hasattr(policy, "base_policy"):
        policy = policy.base_policy
    return isinstance(
        policy, (TrueRewardPerCostPolicy, TrueRewardContextualPolicy, OraclePortfolioPolicy)
    )


def _opportunity_pool(env: BanditEnvironment | StatefulMicrofinanceCMDPEnv) -> str:
    """Record the opening pool before any decision changes affordability."""
    identifiers = (
        (arm.name for arm in env.arms)
        if isinstance(env, BanditEnvironment) and env.mode == BanditMode.ARM
        else (candidate.borrower.id for candidate in env.current_pool)
    )
    pool = zip(identifiers, env.current_groups(), env.current_costs())
    return json.dumps(list(pool), separators=(",", ":"))


def run_episode(
    env: BanditEnvironment,
    policy: BasePolicy,
    target_shares: Mapping[str, float] | None = None,
    seed: int | None = None,
    *,
    parity_tolerance: float | None = None,
    quota_tolerance: float = QUOTA_TOLERANCE,
) -> EpisodeResult:
    """Run a single episode of the environment with the given policy and return the summary and trajectory."""
    targets = dict(target_shares or {})
    arm_mode = env.mode == BanditMode.ARM
    if parity_tolerance is None:
        parity_tolerance = ARM_PARITY_TOLERANCE if arm_mode else CONTEXTUAL_PARITY_TOLERANCE
    env.reset(seed)
    policy.seed = env.config.seed if seed is None else seed
    policy.reset()
    informed = _uses_true_probabilities(policy)
    rows: list[dict[str, float | int | str | None]] = []
    cumulative_reward = 0.0
    cumulative_cost = 0.0
    observed_reward = 0.0
    feedback_at_round_start = ()

    while not env.done:
        round_index = env.round_index
        for feedback in feedback_at_round_start:
            policy.update(feedback)
            observed_reward += feedback.reward
        observed_count = len(feedback_at_round_start)
        row: dict[str, float | int | str | None] = {
            "round": round_index,
            "opportunity_pool": _opportunity_pool(env),
            "selected_index": None,
            "identifier": None,
            "fairness_group": None,
            "reward": 0.0,
            "cost": 0.0,
        }
        if env.feasible_actions():
            action = policy.select_action(env.policy_observation(informed=informed))
            result = env.step(action)
            released = result.observed_feedback
            funded = result.funding_counts
            opportunities = result.opportunity_counts
            population = result.population_opportunity_counts
            cumulative_reward += result.reward
            cumulative_cost += result.cost
            row.update({
                "selected_index": result.selected_index,
                "identifier": result.identifier,
                "fairness_group": result.group,
                "reward": result.reward,
                "cost": result.cost,
            })
            if result.true_repayment_probability is not None:
                row["true_repayment_probability"] = result.true_repayment_probability
            row.update({
                key: value for key, value in result.metadata.items()
                if isinstance(value, (str, int, float))
            })
        else:
            funded = dict(env.group_funding_counts)
            opportunities = dict(env.group_opportunity_counts)
            population = dict(env.group_population_opportunity_counts)
            released = env.advance_round()

        for feedback in released:
            if feedback.due_round == round_index:
                policy.update(feedback)
                observed_reward += feedback.reward
                observed_count += 1
        feedback_at_round_start = tuple(
            feedback for feedback in released if feedback.due_round > round_index
        )
        row.update({
            "cumulative_reward": cumulative_reward,
            "observed_cumulative_reward": observed_reward,
            "cumulative_cost": cumulative_cost,
            "budget_remaining": env.remaining_budget,
            "budget_utilization": budget_utilization(
                env.config.total_budget, env.remaining_budget
            ),
            "reward_per_cost": (
                cumulative_reward / cumulative_cost if cumulative_cost else 0.0
            ),
            "repayment_rate": (
                cumulative_reward / env.total_allocations if env.total_allocations else 0.0
            ),
            **_fairness_metrics(
                funded,
                opportunities,
                None if arm_mode else population,
                targets,
                parity_tolerance,
                quota_tolerance,
            ),
            "observed_feedback_count": observed_count,
        })
        rows.append(row)

    observed_reward += sum(feedback.reward for feedback in feedback_at_round_start)
    trajectory = pd.DataFrame(rows)
    summary: dict[str, float | int | str] = {
        "rounds": env.total_allocations,
        "decision_rounds": env.round_index,
        "cumulative_reward": cumulative_reward,
        "observed_reward_by_horizon": observed_reward,
        "cumulative_cost": cumulative_cost,
        "reward_per_cost": cumulative_reward / cumulative_cost if cumulative_cost else 0.0,
        "repayment_rate": (
            cumulative_reward / env.total_allocations if env.total_allocations else 0.0
        ),
        "budget_utilization": budget_utilization(
            env.config.total_budget, env.remaining_budget
        ),
        **_fairness_metrics(
            env.group_funding_counts,
            env.group_opportunity_counts,
            None if arm_mode else env.group_population_opportunity_counts,
            targets,
            parity_tolerance,
            quota_tolerance,
        ),
        "pending_feedback": len(env.pending_feedback),
    }
    return EpisodeResult(summary=summary, trajectory=trajectory)


EnvironmentFactory = Callable[[int], BanditEnvironment]
PolicyFactory = Callable[[SimpleNamespace, int], BasePolicy]


def _policy_configuration(policy: BasePolicy | CMDPPolicy) -> dict[str, object]:
    """Record constructor settings from the actual policy, including any wrapper."""
    parameters = {}
    for name in signature(type(policy)).parameters:
        if name == "seed" or not hasattr(policy, name):
            continue
        value = getattr(policy, name)
        if isinstance(value, (BasePolicy, CMDPPolicy)):
            value = _policy_configuration(value)
        elif is_dataclass(value):
            value = asdict(value)
        parameters[name] = value
    return {"class": type(policy).__name__, "parameters": parameters}


def compare_policies(
    environment_factory: EnvironmentFactory,
    policy_factories: Mapping[str, PolicyFactory],
    target_shares: Mapping[str, float],
    seeds: Sequence[int],
    *,
    parity_tolerance: float | None = None,
    quota_tolerance: float = QUOTA_TOLERANCE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run episodes for each policy and seed, collecting summaries and trajectories."""
    summaries: list[dict[str, float | int | str]] = []
    trajectories: list[pd.DataFrame] = []
    policy_configurations = {}
    for policy_name, factory in policy_factories.items():
        for seed in seeds:
            env = environment_factory(seed)
            policy = factory(env.policy_observation(), seed)
            if policy_name not in policy_configurations:
                policy_configurations[policy_name] = _policy_configuration(policy)
            episode = run_episode(
                env, policy, target_shares, seed,
                parity_tolerance=parity_tolerance, quota_tolerance=quota_tolerance,
            )
            summary = dict(episode.summary)
            summary.update({"policy": policy_name, "seed": seed})
            summaries.append(summary)
            trajectory = episode.trajectory.copy()
            trajectory["policy"] = policy_name
            trajectory["seed"] = seed
            trajectories.append(trajectory)
    summary_frame = pd.DataFrame(summaries)
    summary_frame.attrs["policy_configurations"] = policy_configurations
    return summary_frame, pd.concat(trajectories, ignore_index=True)


FAIRNESS_METRICS = (
    "demographic_parity_gap",
    "allocation_share_gap",
    "feasible_dp_gap",
    "population_dp_gap",
    "feasible_dp_groups",
    "population_dp_groups",
    "quota_satisfaction_rate",
    "total_quota_deficit",
)

CONSTRAINT_RATE_FIELDS = {
    "parity_satisfaction_rate": "parity_satisfied",
    "quota_constraint_satisfaction_rate": "quota_satisfied",
    "constraint_satisfaction_rate": "constraint_satisfied",
    "population_parity_satisfaction_rate": "population_parity_satisfied",
    "population_constraint_satisfaction_rate": "population_constraint_satisfied",
}

FAIRNESS_COUNT_PREFIXES = ("funded_", "opportunities_", "population_opportunities_")

COMMON_METRICS = (
    "cumulative_reward",
    "observed_reward_by_horizon",
    "cumulative_cost",
    "reward_per_cost",
    "repayment_rate",
    "budget_utilization",
    *FAIRNESS_METRICS,
)


def _aggregate_episode_metrics(
    summaries: pd.DataFrame, metrics: Sequence[str],
) -> pd.DataFrame:
    metrics = [metric for metric in metrics if metric in summaries]
    counts = [column for column in summaries if column.startswith(FAIRNESS_COUNT_PREFIXES)]
    data = summaries.copy()
    data[counts] = data[counts].fillna(0)
    grouped = data.groupby("policy")
    records = {}
    for policy, frame in grouped:
        fields = {}
        for metric in dict.fromkeys([*metrics, *counts]):
            fields.update({(metric, key): value for key, value in describe(frame[metric]).items()})
        if "parity_evaluable" in frame:
            fields["episodes_evaluated", ""] = len(frame)
            for evaluable in ("parity_evaluable", "population_parity_evaluable"):
                if evaluable in frame:
                    fields[f"{evaluable}_episodes", ""] = int(frame[evaluable].sum())
            for rate, outcome in CONSTRAINT_RATE_FIELDS.items():
                if outcome in frame:
                    fields.update({
                        (rate, "" if key == "mean" else key): value
                        for key, value in describe(frame[outcome].fillna(0), proportion=True).items()
                    })
        records[policy] = fields
    aggregate = pd.DataFrame.from_dict(records, orient="index").rename_axis("policy")
    aggregate.columns = pd.MultiIndex.from_tuples(aggregate.columns)
    return aggregate.sort_values(("cumulative_reward", "mean"), ascending=False)


def aggregate_results(summaries: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the results by policy, computing the mean and standard deviation for each common metric."""
    return _aggregate_episode_metrics(summaries, COMMON_METRICS)


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
    paired_comparisons(summaries, mode, sorted(summaries["seed"].unique())).to_csv(
        output / "paired_comparisons.csv", index=False
    )
    for metric in (
        "cumulative_reward",
        "repayment_rate",
        "allocation_share_gap" if mode == "arm" else "feasible_dp_gap",
        "population_dp_gap",
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
        display = aggregate.loc[:, aggregate.columns.get_level_values(1).isin(["mean", "std"])]
        print(display.round(5).to_string())
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
    *,
    quota_tolerance: float = QUOTA_TOLERANCE,
) -> CMDPEpisodeResult:
    """Run one stateful Phase 3 lending episode."""

    env.reset(seed)
    policy.seed = seed
    policy.reset()
    informed = _uses_true_probabilities(policy)
    rows: list[dict[str, float | int | str]] = []

    while not env.done:
        matured_now = env.matured_at_round_start
        for feedback in matured_now:
            policy.update(
                feedback.learning_context,
                feedback.reward,
                feedback.principal,
            )

        opportunity_pool = _opportunity_pool(env)
        funded_this_round = 0
        matured_this_round = len(matured_now)
        hidden_reward_this_round = 0.0
        disbursed_this_round = 0.0
        while funded_this_round < env.config.max_loans_per_round and env.feasible_actions():
            observation = env.policy_observation(informed=informed)
            action = policy.select_action(observation)
            if action is None:
                break
            learning_context = policy.decision_context(observation, action)
            result = env.fund(action, learning_context)
            for feedback in result.observed_feedback:
                policy.update(feedback.learning_context, feedback.reward, feedback.principal)
            funded_this_round += 1
            matured_this_round += len(result.observed_feedback)
            hidden_reward_this_round += result.hidden_reward
            disbursed_this_round += result.principal

        lambda_quota, lambda_parity = _policy_lambdas(policy)
        rows.append(
            {
                "round": env.round_index,
                "opportunity_pool": opportunity_pool,
                "funded_this_round": funded_this_round,
                "matured_this_round": matured_this_round,
                "hidden_reward_funded_this_round": hidden_reward_this_round,
                "disbursed_this_round": disbursed_this_round,
                "cumulative_funded": env.total_allocations,
                "eventual_cumulative_reward": env.eventual_reward(),
                "observed_cumulative_reward": env.observed_repayments,
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
                "cumulative_unrecycled_successful_principal": (
                    env.cumulative_unrecycled_successful_principal
                ),
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
                **_fairness_metrics(
                    env.group_funding_counts,
                    env.group_opportunity_counts,
                    env.group_population_opportunity_counts,
                    target_shares,
                    parity_tolerance,
                    quota_tolerance,
                ),
                "lambda_quota": lambda_quota,
                "lambda_parity": lambda_parity,
                "state_bucket": env.state_bucket(
                    target_shares,
                    parity_tolerance,
                ),
            }
        )
        policy.on_round_end(env.policy_observation(informed=informed))
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
        "unrecycled_successful_principal_by_horizon": (
            env.cumulative_unrecycled_successful_principal
        ),
        "final_available_capital": env.available_capital,
        "final_capital_at_risk": env.capital_at_risk,
        "eventual_terminal_capital": eventual_terminal_capital,
        "capital_survival_ratio": capital_survival_ratio(
            env.available_capital,
            env.capital_at_risk,
            env.config.initial_capital,
        ),
        **_fairness_metrics(
            env.group_funding_counts,
            env.group_opportunity_counts,
            env.group_population_opportunity_counts,
            target_shares,
            parity_tolerance,
            quota_tolerance,
        ),
        "pending_loans": len(env.pending_loans),
        "lambda_quota_final": lambda_quota,
        "lambda_parity_final": lambda_parity,
    }

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
    *,
    quota_tolerance: float = QUOTA_TOLERANCE,
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
    policy_configurations = {}

    for policy_name, factory in policy_factories.items():
        for seed in seeds:
            env = StatefulMicrofinanceCMDPEnv(env_config, shared_encoder)
            policy = factory(shared_encoder.n_features, seed)
            if policy_name not in policy_configurations:
                policy_configurations[policy_name] = _policy_configuration(policy)
            episode = run_cmdp_episode(
                env,
                policy,
                target_shares,
                parity_tolerance,
                seed,
                quota_tolerance=quota_tolerance,
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

    summary_frame = pd.DataFrame(summaries)
    summary_frame.attrs["policy_configurations"] = policy_configurations
    return (
        summary_frame,
        concat_or_empty(trajectories),
        concat_or_empty(funding_logs),
        concat_or_empty(maturity_logs),
    )


CMDP_METRICS = (
    "loans_funded",
    "cumulative_reward",
    "observed_reward_by_horizon",
    "repayment_rate",
    "cumulative_disbursed",
    "reward_per_cost",
    "capital_turnover",
    "cumulative_recycled",
    "default_loss_by_horizon",
    "unrecycled_successful_principal_by_horizon",
    "final_available_capital",
    "final_capital_at_risk",
    "eventual_terminal_capital",
    "capital_survival_ratio",
    *FAIRNESS_METRICS,
    "lambda_quota_final",
    "lambda_parity_final",
)


def aggregate_cmdp_results(summaries: pd.DataFrame) -> pd.DataFrame:
    return _aggregate_episode_metrics(summaries, CMDP_METRICS)


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
    paired_comparisons(summaries, "cmdp", sorted(summaries["seed"].unique())).to_csv(
        output / "paired_comparisons.csv", index=False
    )
    for metric in (
        "eventual_cumulative_reward",
        "available_capital",
        "capital_at_risk",
        "capital_turnover",
        "feasible_dp_gap",
        "population_dp_gap",
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


def flatten_aggregate(aggregate: pd.DataFrame) -> pd.DataFrame:
    flattened = aggregate.copy()
    if isinstance(flattened.columns, pd.MultiIndex):
        flattened.columns = [
            "_".join(
                str(part) for part in column
                if part and not str(part).startswith("Unnamed: ")
            )
            for column in flattened.columns
        ]
    flattened = flattened.reset_index()
    if "policy" not in flattened and len(flattened.columns):
        flattened = flattened.rename(columns={flattened.columns[0]: "policy"})
    return flattened


def rank_results(
    aggregate: pd.DataFrame,
    parity_tolerance: float,
    quota_tolerance: float = QUOTA_TOLERANCE,
) -> pd.DataFrame:
    """Rank policies by fairness feasibility, reward efficiency, and reward."""

    if not 0.0 <= parity_tolerance <= 1.0:
        raise ValueError("parity_tolerance must be in [0, 1]")
    if quota_tolerance < 0.0:
        raise ValueError("quota_tolerance cannot be negative")
    ranked = flatten_aggregate(aggregate)
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
    ranked["mean_thresholds_met"] = ranked["fairness_violation"].eq(0)
    if "constraint_satisfaction_rate" in ranked:
        ranked["fairness_feasible"] = ranked["constraint_satisfaction_rate"].eq(1)
        ranked["_compliance_order"] = -ranked["constraint_satisfaction_rate"]
    else:
        # Old aggregate files contain only means, so episode compliance is unknown.
        ranked["fairness_feasible"] = pd.NA
        ranked["_compliance_order"] = 0.0
    ranked["_violation_order"] = ranked["fairness_violation"].where(
        ~ranked["fairness_feasible"].eq(True).fillna(False), 0.0
    )
    ranked = ranked.sort_values(
        [
            "_compliance_order",
            "_violation_order",
            "reward_per_cost_mean",
            "cumulative_reward_mean",
            "policy",
        ],
        ascending=[True, True, False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1, dtype=int))
    return ranked.drop(columns=["_compliance_order", "_violation_order"])


def winner_summary(ranked: pd.DataFrame, mode: str) -> str:
    """Describe the first-ranked policy for this configuration."""

    if ranked.empty:
        return f"{mode.title()} has no policies to rank."
    winner = ranked.iloc[0]
    label = "Phase 2 contextual" if mode == "contextual" else "Phase 3 portfolio"
    reward_per_cost = float(winner["reward_per_cost_mean"])
    cumulative_reward = float(winner["cumulative_reward_mean"])
    parity_gap = winner["demographic_parity_gap_mean"]
    gap_label = f"{parity_gap:.6f}" if pd.notna(parity_gap) else "undefined"
    quota_deficit = float(winner["total_quota_deficit_mean"])
    if "constraint_satisfaction_rate" in ranked:
        satisfied = winner["constraint_satisfaction_rate"]
        qualification = (
            f"It met both feasible-opportunity constraints in {satisfied:.1%} of episodes"
        )
    else:
        qualification = "It led the mean-based ranking. Episode compliance is unavailable"
    return (
        f"{label} ranking leader: {winner['policy']}. {qualification}, with mean "
        f"reward-per-cost {reward_per_cost:.6f}, eventual successful repayments "
        f"{cumulative_reward:.3f}, feasible parity gap {gap_label}, and total "
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
        "constraint_satisfaction_rate": "Feasible Joint Rate",
        "population_constraint_satisfaction_rate": "Population Joint Rate",
        "cumulative_reward_mean": "Eventual Successes",
        "reward_per_cost_mean": "Reward/Cost",
        "repayment_rate_mean": "Repayment Rate",
        "demographic_parity_gap_mean": "Feasible DP Gap",
        "population_dp_gap_mean": "Population DP Gap",
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
    for package in ("numpy", "pandas", "matplotlib", "scipy"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def episode_completion(expected: int, actual: int) -> dict[str, object]:
    return {
        "expected_episodes": expected,
        "actual_episodes": actual,
        "status": "complete" if actual == expected else "incomplete",
    }


def reproducibility_metadata() -> dict[str, object]:
    """Metadata shared by standard evaluations and robustness sweeps."""
    def git_output(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", "--no-optional-locks", *args],
                cwd=Path(__file__).resolve().parent,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=5,
            ).strip()
        except (OSError, subprocess.SubprocessError):
            return None

    status = git_output("status", "--porcelain")
    return {
        "simulation_version": 2,
        "audit_format": 1,
        "confidence_intervals": CI_METHODS,
        "rng_convention": {
            "generator": "NumPy default_rng (PCG64)",
            "seed_derivation": "SeedSequence(episode seed), children 0, 1 and 2",
            "opportunities": "Child 0 generates costs or applicant pools in round order",
            "potential_outcomes": (
                "Child 1 draws a uniform per arm or applicant in original round order; "
                "repayment occurs when the draw is below the true probability"
            ),
            "policy": "Child 2 supplies exploration, sampling and tie-breaking",
        },
        "package_versions": _package_versions(),
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
    }


def _mean_std_column(ranked: pd.DataFrame, metric: str) -> pd.Series:
    means = ranked[f"{metric}_mean"]
    standard_deviations = ranked.get(
        f"{metric}_std", pd.Series(0.0, index=ranked.index)
    )
    return pd.Series(
        [
            f"{float(mean):.6f} ± {float(std):.6f}" if pd.notna(mean) else "Undefined"
            for mean, std in zip(means, standard_deviations)
        ],
        index=ranked.index,
    )


def _html_ranked_table(ranked: pd.DataFrame, mode: str) -> str:
    display = pd.DataFrame(
        {
            "Rank": ranked["rank"],
            "Policy": ranked["policy"],
            "Eventual successful repayments (mean ± std)": _mean_std_column(
                ranked, "cumulative_reward"
            ),
            "Reward/cost (mean ± std)": _mean_std_column(
                ranked, "reward_per_cost"
            ),
            "Repayment rate (mean ± std)": _mean_std_column(
                ranked, "repayment_rate"
            ),
            "Feasible DP gap (mean ± std)": _mean_std_column(
                ranked, "demographic_parity_gap"
            ),
            "Quota deficit (mean ± std)": _mean_std_column(
                ranked, "total_quota_deficit"
            ),
        }
    )
    if "population_dp_gap_mean" in ranked:
        display["Population DP gap (mean ± std)"] = _mean_std_column(
            ranked, "population_dp_gap"
        )
    for column, label in (
        ("parity_satisfaction_rate", "Feasible parity"),
        ("population_parity_satisfaction_rate", "Population parity"),
        ("quota_constraint_satisfaction_rate", "Quota constraint"),
        ("constraint_satisfaction_rate", "Feasible joint"),
        ("population_constraint_satisfaction_rate", "Population joint"),
    ):
        if column in ranked:
            display[label + " satisfaction"] = ranked[column].map("{:.1%}".format)
    for column, label in (
        ("episodes_evaluated", "Episodes"),
        ("parity_evaluable_episodes", "Feasible parity evaluable"),
        ("population_parity_evaluable_episodes", "Population parity evaluable"),
    ):
        if column in ranked:
            display[label] = ranked[column]
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
    quota_tolerance: float = QUOTA_TOLERANCE,
    *,
    policy_configurations: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    """Rank results and persist reproducibility and human-readable reports."""

    output.mkdir(parents=True, exist_ok=True)
    ranked = rank_results(aggregate, parity_tolerance, quota_tolerance)
    summary = winner_summary(ranked, mode)
    ranked.to_csv(output / "ranked_results.csv", index=False)
    (output / "winner_summary.txt").write_text(summary + "\n", encoding="utf-8")
    episode_count = ("episodes_evaluated", "")
    completion = (
        episode_completion(len(seeds) * len(ranked), int(aggregate[episode_count].sum()))
        if episode_count in aggregate else {}
    )
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
        "policy_configurations": dict(policy_configurations or {}),
        **completion,
        **reproducibility_metadata(),
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    title = "Phase 2 Contextual Bandits" if mode == "contextual" else "Phase 3 Portfolio Policies"
    benchmark_description = (
        "The informed contextual benchmark greedily scores true repayment probability per unit principal."
        if mode == "contextual"
        else "The informed portfolio heuristic subtracts a liquidity-dependent cost penalty from true repayment probability."
    )
    report = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(title)} evaluation</title><style>{REPORT_STYLE}</style></head>
<body><h1>{escape(title)} evaluation</h1>
<p class="winner">{escape(summary)}</p>
<p class="note">Episode thresholds are {parity_tolerance:.6f} for parity and {quota_tolerance:.6f} for total quota deficit. Satisfaction rates use all evaluated episodes, with parity evaluability shown separately. Ranking uses feasible-opportunity joint satisfaction first, then mean threshold excess among policies with incomplete compliance, then reward-per-cost and cumulative reward.</p>
<p class="note">{escape(benchmark_description)} Ranking describes this simulated configuration.</p>
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
        arm = flatten_aggregate(arm_aggregate)
        arm = arm.sort_values("cumulative_reward_mean", ascending=False)
        columns = [
            column
            for column in (
                "policy",
                "cumulative_reward_mean",
                "reward_per_cost_mean",
                "repayment_rate_mean",
                "allocation_share_gap_mean",
                "quota_satisfaction_rate_mean",
            )
            if column in arm
        ]
        arm_table = arm[columns].rename(
            columns={
                "policy": "Policy",
                "cumulative_reward_mean": "Mean successful repayments",
                "reward_per_cost_mean": "Mean reward/cost",
                "repayment_rate_mean": "Mean repayment rate",
                "allocation_share_gap_mean": "Mean allocation-share gap",
                "quota_satisfaction_rate_mean": "Mean fraction of quota targets met",
            }
        )
        sections.append(
            "<h2>Mode 1 — Arm Bandits</h2>"
            '<p class="note">Policies are ordered by mean cumulative reward; '
            "the informed benchmark greedily scores true repayment probability per unit principal.</p>"
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
            else "Mode 3 — Phase 3 Portfolio Policies"
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
