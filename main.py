from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from Algorithms.bandit import RandomPolicy
from Algorithms.epsilon_first import EpsilonFirstPolicy
from Algorithms.epsilon_greedy import EpsilonGreedyPolicy
from Algorithms.fairness_constrained import FairnessConstrainedPolicy
from Algorithms.fairness_penalty import FairnessPenaltyPolicy
from Algorithms.fractional_kube import FractionalKUBEPolicy, OracleRewardPerCostPolicy
from Algorithms.kube import KUBEPolicy
from Algorithms.thompson_sampling import BudgetedUCBPolicy, ThompsonSamplingPolicy
from Algorithms.ucb import UCBPolicy
from evaluate import compare_policies
from models.arm_spec import ArmSpec
from models.environment_config import EnvironmentConfig


def build_policy_factories(arms: list[ArmSpec], target_shares: dict[str, float]):
    arm_groups = [arm.group for arm in arms]

    return {
        "random": lambda n, seed: RandomPolicy(n, seed),
        "epsilon_greedy": lambda n, seed: EpsilonGreedyPolicy(
            n, epsilon=0.10, seed=seed
        ),
        "ucb": lambda n, seed: UCBPolicy(n, seed=seed),
        "thompson": lambda n, seed: ThompsonSamplingPolicy(n, seed=seed),
        "epsilon_first": lambda n, seed: EpsilonFirstPolicy(
            n, exploration_budget_fraction=0.20, seed=seed
        ),
        "budgeted_ucb": lambda n, seed: BudgetedUCBPolicy(n, seed=seed),
        "fractional_kube": lambda n, seed: FractionalKUBEPolicy(n, seed=seed),
        "kube": lambda n, seed: KUBEPolicy(n, seed=seed),
        "fair_ucb_hard": lambda n, seed: FairnessConstrainedPolicy(
            base_policy=BudgetedUCBPolicy(n, seed=seed),
            arm_groups=arm_groups,
            target_shares=target_shares,
            parity_tolerance=0.35,
            seed=seed,
        ),
        "fair_ucb_soft": lambda n, seed: FairnessPenaltyPolicy(
            base_policy=BudgetedUCBPolicy(n, seed=seed),
            arm_groups=arm_groups,
            target_shares=target_shares,
            penalty_lambda=2.0,
            seed=seed,
        ),
        "oracle_ratio": lambda n, seed: OracleRewardPerCostPolicy(n, seed=seed),
    }


def aggregate_results(summaries: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "cumulative_reward",
        "reward_per_cost",
        "budget_utilization",
        "demographic_parity_gap",
        "quota_satisfaction_rate",
    ]
    return (
        summaries.groupby("policy")[metrics]
        .agg(["mean", "std"])
        .sort_values(("cumulative_reward", "mean"), ascending=False)
    )


def plot_metric_over_time(
    trajectories: pd.DataFrame,
    metric: str,
    output_path: Path,
) -> None:
    averaged = trajectories.groupby(["policy", "round"], as_index=False)[metric].mean()
    fig, ax = plt.subplots(figsize=(10, 6))
    for policy, frame in averaged.groupby("policy"):
        ax.plot(frame["round"], frame[metric], label=policy)
    ax.set_xlabel("Round")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Phase 1: {metric.replace('_', ' ').title()} over time")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase-1 bandit experiments")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--output", type=Path, default=os.getcwd() + "/outputs")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    # The simple three-group example from the project plan.
    arms = [
        ArmSpec("Group A", "A", repayment_probability=0.80, cost_mean=12.0),
        ArmSpec("Group B", "B", repayment_probability=0.65, cost_mean=8.0),
        ArmSpec("Group C", "C", repayment_probability=0.50, cost_mean=4.0),
    ]
    env_config = EnvironmentConfig(total_budget=600.0, max_rounds=150)
    target_shares = {"A": 0.20, "B": 0.30, "C": 0.30}
    seeds = list(range(args.seeds))

    summaries, trajectories = compare_policies(
        arms=arms,
        env_config=env_config,
        policy_factories=build_policy_factories(arms, target_shares),
        target_shares=target_shares,
        seeds=seeds,
    )

    summaries.to_csv(args.output / "phase1_episode_summaries.csv", index=False)
    trajectories.to_csv(args.output / "phase1_trajectories.csv", index=False)
    aggregate = aggregate_results(summaries)
    aggregate.to_csv(args.output / "phase1_aggregate_results.csv")

    plot_metric_over_time(
        trajectories,
        "cumulative_reward",
        args.output / "cumulative_reward.png",
    )
    plot_metric_over_time(
        trajectories,
        "demographic_parity_gap",
        args.output / "demographic_parity_gap.png",
    )
    plot_metric_over_time(
        trajectories,
        "quota_satisfaction_rate",
        args.output / "quota_satisfaction_rate.png",
    )

    print("\nAggregate results:\n")
    print(aggregate.round(4).to_string())
    print(f"\nSaved outputs to: {args.output}")


if __name__ == "__main__":
    main()
