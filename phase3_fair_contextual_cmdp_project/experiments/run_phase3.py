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
from phase3.models import CMDPEnvironmentConfig, LagrangianConfig
from phase3.policies import (
    ContextOnlyBudgetedLinUCBPolicy,
    HardFairCMDPPolicy,
    LagrangianFairCMDPPolicy,
    OraclePortfolioPolicy,
    PortfolioAwareLinUCBPolicy,
    PortfolioAwareThompsonPolicy,
    RandomCMDPPolicy,
)
from phase3.runner import compare_policies


def build_policy_factories(
    target_shares: dict[str, float],
    parity_tolerance: float,
):
    lagrangian_config = LagrangianConfig(
        target_shares=target_shares,
        parity_tolerance=parity_tolerance,
        quota_tolerance=0.01,
        dual_learning_rate=0.20,
        initial_lambda_quota=0.20,
        initial_lambda_parity=0.20,
        max_lambda=8.0,
        warmup_rounds=5,
    )
    return {
        "random_cmdp": lambda d, seed: RandomCMDPPolicy(d, seed=seed),
        # Phase-2-style control: context + cost, but no portfolio state.
        "context_only_budgeted_linucb": lambda d, seed: ContextOnlyBudgetedLinUCBPolicy(
            d, alpha=0.70, seed=seed
        ),
        # Phase-3 state-aware policies.
        "portfolio_linucb": lambda d, seed: PortfolioAwareLinUCBPolicy(
            d, alpha=0.70, base_shadow_price=0.22, seed=seed
        ),
        "portfolio_thompson": lambda d, seed: PortfolioAwareThompsonPolicy(
            d, posterior_scale=0.30, base_shadow_price=0.22, seed=seed
        ),
        "cmdp_fair_hard": lambda d, seed: HardFairCMDPPolicy(
            base_policy=PortfolioAwareLinUCBPolicy(
                d, alpha=0.70, base_shadow_price=0.22, seed=seed
            ),
            target_shares=target_shares,
            parity_tolerance=parity_tolerance,
            seed=seed,
        ),
        "cmdp_fair_lagrangian": lambda d, seed: LagrangianFairCMDPPolicy(
            base_policy=PortfolioAwareLinUCBPolicy(
                d, alpha=0.70, base_shadow_price=0.22, seed=seed
            ),
            config=lagrangian_config,
            seed=seed,
        ),
        "oracle_portfolio": lambda d, seed: OraclePortfolioPolicy(
            d, state_aware=True, seed=seed
        ),
    }


def aggregate_results(summaries: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "cumulative_reward",
        "repayment_rate",
        "reward_per_cost",
        "capital_turnover",
        "cumulative_recycled",
        "default_loss_by_horizon",
        "eventual_terminal_capital",
        "demographic_parity_gap",
        "quota_satisfaction_rate",
        "total_quota_deficit",
    ]
    return (
        summaries.groupby("policy")[metrics]
        .agg(["mean", "std"])
        .sort_values(("cumulative_reward", "mean"), ascending=False)
    )


def plot_metric(trajectories: pd.DataFrame, metric: str, output: Path) -> None:
    averaged = trajectories.groupby(["policy", "round"], as_index=False)[metric].mean()
    fig, ax = plt.subplots(figsize=(10, 6))
    for policy, frame in averaged.groupby("policy"):
        ax.plot(frame["round"], frame[metric], label=policy)
    ax.set_xlabel("Round")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Phase 3: {metric.replace('_', ' ').title()} over time")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Phase-3 contextual CMDP experiments with delayed repayment and capital recycling"
    )
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--pool-size", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--capital", type=float, default=600_000.0)
    parser.add_argument("--delay", type=int, default=8)
    parser.add_argument("--recycle-fraction", type=float, default=1.0)
    parser.add_argument("--default-recovery", type=float, default=0.0)
    parser.add_argument("--parity-tolerance", type=float, default=0.05)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs_phase3")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    target_shares = {
        "disabled": 0.04,
        "reserved": 0.08,
        "minority": 0.12,
        "women": 0.25,
    }
    encoder = BorrowerFeatureEncoder(include_protected_attributes=False)
    config = CMDPEnvironmentConfig(
        initial_capital=args.capital,
        max_rounds=args.rounds,
        pool_size=args.pool_size,
        max_loans_per_round=args.batch_size,
        repayment_delay_rounds=args.delay,
        recycle_fraction=args.recycle_fraction,
        default_recovery_fraction=args.default_recovery,
    )

    summaries, trajectories, funding, maturities = compare_policies(
        env_config=config,
        policy_factories=build_policy_factories(target_shares, args.parity_tolerance),
        target_shares=target_shares,
        parity_tolerance=args.parity_tolerance,
        seeds=list(range(args.seeds)),
        encoder=encoder,
    )

    summaries.to_csv(args.output / "phase3_episode_summaries.csv", index=False)
    trajectories.to_csv(args.output / "phase3_trajectories.csv", index=False)
    funding.to_csv(args.output / "phase3_funding_log.csv", index=False)
    maturities.to_csv(args.output / "phase3_maturity_log.csv", index=False)
    aggregate = aggregate_results(summaries)
    aggregate.to_csv(args.output / "phase3_aggregate_results.csv")

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
        plot_metric(trajectories, metric, args.output / f"{metric}.png")

    print("\nPhase 3 aggregate results:\n")
    print(aggregate.round(5).to_string())
    print(f"\nSaved outputs to: {args.output}")


if __name__ == "__main__":
    main()
