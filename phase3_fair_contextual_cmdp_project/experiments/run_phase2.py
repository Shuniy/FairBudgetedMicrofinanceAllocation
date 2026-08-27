from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phase2.feature_encoder import BorrowerFeatureEncoder
from phase2.models import ContextualEnvironmentConfig
from phase2.policies import (
    ContextualEpsilonGreedyPolicy,
    ContextualThompsonSamplingPolicy,
    HardFairContextualPolicy,
    LinUCBPolicy,
    OracleContextualPolicy,
    RandomContextualPolicy,
    SoftFairContextualPolicy,
)
from phase2.runner import compare_policies


def build_policy_factories(target_shares: dict[str, float]):
    return {
        "random": lambda d, seed: RandomContextualPolicy(d, seed=seed),
        "contextual_epsilon_greedy": lambda d, seed: ContextualEpsilonGreedyPolicy(
            d, epsilon=0.10, budget_aware=False, seed=seed
        ),
        "contextual_epsilon_greedy_ratio": lambda d, seed: ContextualEpsilonGreedyPolicy(
            d, epsilon=0.10, budget_aware=True, seed=seed
        ),
        "linucb": lambda d, seed: LinUCBPolicy(
            d, alpha=0.70, budget_aware=False, seed=seed
        ),
        "budgeted_linucb": lambda d, seed: LinUCBPolicy(
            d, alpha=0.70, budget_aware=True, seed=seed
        ),
        "contextual_thompson": lambda d, seed: ContextualThompsonSamplingPolicy(
            d, posterior_scale=0.30, budget_aware=False, seed=seed
        ),
        "budgeted_contextual_thompson": lambda d, seed: ContextualThompsonSamplingPolicy(
            d, posterior_scale=0.30, budget_aware=True, seed=seed
        ),
        "fair_linucb_hard": lambda d, seed: HardFairContextualPolicy(
            base_policy=LinUCBPolicy(
                d, alpha=0.70, budget_aware=True, seed=seed
            ),
            target_shares=target_shares,
            parity_tolerance=0.010,
            seed=seed,
        ),
        "fair_linucb_soft": lambda d, seed: SoftFairContextualPolicy(
            base_policy=LinUCBPolicy(
                d, alpha=0.70, budget_aware=True, seed=seed
            ),
            target_shares=target_shares,
            penalty_lambda=0.015,
            seed=seed,
        ),
        "oracle_contextual_ratio": lambda d, seed: OracleContextualPolicy(
            d, budget_aware=True, seed=seed
        ),
    }


def aggregate_results(summaries: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "cumulative_reward",
        "reward_per_cost",
        "repayment_rate",
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
    ax.set_title(f"Phase 2: {metric.replace('_', ' ').title()} over time")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase-2 contextual-bandit experiments")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=150)
    parser.add_argument("--pool-size", type=int, default=20)
    parser.add_argument("--budget", type=float, default=600_000.0)
    parser.add_argument("--feedback-delay", type=int, default=0)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs_phase2")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    encoder = BorrowerFeatureEncoder(include_protected_attributes=False)
    config = ContextualEnvironmentConfig(
        total_budget=args.budget,
        max_rounds=args.rounds,
        pool_size=args.pool_size,
        feedback_delay_rounds=args.feedback_delay,
    )

    # Minimum shares apply to the mutually-exclusive primary fairness groups.
    target_shares = {
        "disabled": 0.04,
        "reserved": 0.08,
        "minority": 0.12,
        "women": 0.25,
    }

    summaries, trajectories = compare_policies(
        env_config=config,
        policy_factories=build_policy_factories(target_shares),
        target_shares=target_shares,
        seeds=list(range(args.seeds)),
        encoder=encoder,
    )

    summaries.to_csv(args.output / "phase2_episode_summaries.csv", index=False)
    trajectories.to_csv(args.output / "phase2_trajectories.csv", index=False)
    aggregate = aggregate_results(summaries)
    aggregate.to_csv(args.output / "phase2_aggregate_results.csv")

    for metric in (
        "cumulative_reward",
        "repayment_rate",
        "demographic_parity_gap",
        "quota_satisfaction_rate",
    ):
        plot_metric_over_time(
            trajectories,
            metric,
            args.output / f"{metric}.png",
        )

    print("\nPhase 2 aggregate results:\n")
    print(aggregate.round(5).to_string())
    print(f"\nFeature count: {encoder.n_features}")
    print(f"Saved outputs to: {args.output}")


if __name__ == "__main__":
    main()
