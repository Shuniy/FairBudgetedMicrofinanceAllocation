from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path

from Algorithms.bandit import RandomPolicy
from Algorithms.budgeted_ucb import BudgetedUCBPolicy
from Algorithms.cmdp import (
    ContextOnlyBudgetedLinUCBPolicy,
    HardFairCMDPPolicy,
    LagrangianFairCMDPPolicy,
    OraclePortfolioPolicy,
    PortfolioAwareLinUCBPolicy,
    PortfolioAwareThompsonPolicy,
    RandomCMDPPolicy,
)
from Algorithms.contextual import (
    ContextualEpsilonGreedyPolicy,
    ContextualThompsonSamplingPolicy,
    LinUCBPolicy,
    TrueRewardContextualPolicy,
    RandomContextualPolicy,
)
from Algorithms.epsilon_first import EpsilonFirstPolicy
from Algorithms.epsilon_greedy import (
    DecayingEpsilonGreedyPolicy,
    EpsilonGreedyPolicy,
)
from Algorithms.fairness import HardFairPolicy, SoftFairPolicy
from Algorithms.fractional_kube import FractionalKUBEPolicy
from Algorithms.kube import KUBEPolicy
from Algorithms.reward_per_cost import RewardPerCostPolicy
from Algorithms.thompson_sampling import ThompsonSamplingPolicy
from Algorithms.true_repayment_cost import TrueRewardPerCostPolicy
from Algorithms.ucb import UCBPolicy
from Data.arm_data import default_arm_configuration, load_arm_configuration
from Data.feature_encoder import BorrowerFeatureEncoder
from Environments.base_environment import BanditEnvironment
from evaluate import (
    compare_cmdp_policies,
    compare_policies,
    display_results,
    save_cmdp_results,
    save_results,
)
from models.environment_config import (
    CMDPEnvironmentConfig,
    ContextualEnvironmentConfig,
    EnvironmentConfig,
)
from models.fairness_config import LagrangianConfig


MODE_ALIASES = {
    "1": "arm",
    "2": "contextual",
    "3": "cmdp",
    "4": "all",
    "arm": "arm",
    "contextual": "contextual",
    "cmdp": "cmdp",
    "all": "all",
}

DEFAULT_TARGET_SHARES = {
    "disabled": 0.04,
    "reserved": 0.08,
    "minority": 0.12,
    "women": 0.25,
}


def build_arm_policy_factories(target_shares: dict[str, float]):
    """Builds a dictionary of arm policy factories based on the provided target shares.
    Each factory is a lambda function that takes an environment and a seed, and returns an instance of the corresponding policy."""
    def n_arms(env: BanditEnvironment) -> int:
        if env.mode != "arm":
            raise TypeError("arm policy requires an arm environment")
        return env.n_arms

    return {
        "random": lambda env, seed: RandomPolicy(n_arms(env), seed),
        "epsilon_greedy": lambda env, seed: EpsilonGreedyPolicy(
            n_arms(env), epsilon=0.10, seed=seed
        ),
        "decaying_epsilon_greedy": lambda env, seed: DecayingEpsilonGreedyPolicy(
            n_arms(env), seed=seed
        ),
        "ucb": lambda env, seed: UCBPolicy(n_arms(env), seed=seed),
        "thompson": lambda env, seed: ThompsonSamplingPolicy(n_arms(env), seed=seed),
        "epsilon_first": lambda env, seed: EpsilonFirstPolicy(
            n_arms(env), exploration_budget_fraction=0.20, seed=seed
        ),
        "reward_per_cost": lambda env, seed: RewardPerCostPolicy(
            n_arms(env), seed=seed
        ),
        "budgeted_ucb": lambda env, seed: BudgetedUCBPolicy(
            n_arms(env), seed=seed
        ),
        "fractional_kube": lambda env, seed: FractionalKUBEPolicy(
            n_arms(env), seed=seed
        ),
        "kube": lambda env, seed: KUBEPolicy(n_arms(env), seed=seed),
        "fair_ucb_hard": lambda env, seed: HardFairPolicy(
            BudgetedUCBPolicy(n_arms(env), seed=seed),
            target_shares,
            parity_tolerance=0.35,
            seed=seed,
        ),
        "fair_ucb_soft": lambda env, seed: SoftFairPolicy(
            BudgetedUCBPolicy(n_arms(env), seed=seed),
            target_shares,
            penalty_lambda=2.0,
            seed=seed,
        ),
        "true_reward_per_cost": lambda env, seed: TrueRewardPerCostPolicy(
            n_arms(env), seed=seed
        ),
    }


def build_contextual_policy_factories(target_shares: dict[str, float]):
    """Builds a dictionary of contextual policy factories based on the provided target shares.
    Each factory is a lambda function that takes an environment and a seed, and returns an instance of the corresponding policy."""
    def dimensions(env: BanditEnvironment) -> int:
        if env.mode != "contextual":
            raise TypeError("contextual policy requires a contextual environment")
        return env.n_features

    return {
        "random": lambda env, seed: RandomContextualPolicy(dimensions(env), seed),
        "contextual_epsilon_greedy": lambda env, seed: ContextualEpsilonGreedyPolicy(
            dimensions(env), epsilon=0.10, seed=seed
        ),
        "budgeted_contextual_epsilon_greedy": lambda env, seed: ContextualEpsilonGreedyPolicy(
            dimensions(env), epsilon=0.10, budget_aware=True, seed=seed
        ),
        "linucb": lambda env, seed: LinUCBPolicy(
            dimensions(env), alpha=0.70, seed=seed
        ),
        "budgeted_linucb": lambda env, seed: LinUCBPolicy(
            dimensions(env), alpha=0.70, budget_aware=True, seed=seed
        ),
        "contextual_thompson": lambda env, seed: ContextualThompsonSamplingPolicy(
            dimensions(env), posterior_scale=0.30, seed=seed
        ),
        "budgeted_contextual_thompson": lambda env, seed: ContextualThompsonSamplingPolicy(
            dimensions(env), posterior_scale=0.30, budget_aware=True, seed=seed
        ),
        "fair_linucb_hard": lambda env, seed: HardFairPolicy(
            LinUCBPolicy(
                dimensions(env), alpha=0.70, budget_aware=True, seed=seed
            ),
            target_shares,
            parity_tolerance=0.010,
            seed=seed,
        ),
        "fair_linucb_soft": lambda env, seed: SoftFairPolicy(
            LinUCBPolicy(
                dimensions(env), alpha=0.70, budget_aware=True, seed=seed
            ),
            target_shares,
            penalty_lambda=0.015,
            seed=seed,
        ),
        "true_contextual_reward_per_cost": lambda env, seed: TrueRewardContextualPolicy(
            dimensions(env), budget_aware=True, seed=seed
        ),
    }


def build_cmdp_policy_factories(
    target_shares: Mapping[str, float],
    parity_tolerance: float,
):
    """Build Phase 3 policies while reusing the parent contextual ridge model."""

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
        "random_cmdp": lambda dimensions, seed: RandomCMDPPolicy(
            dimensions, seed=seed
        ),
        "context_only_budgeted_linucb": (
            lambda dimensions, seed: ContextOnlyBudgetedLinUCBPolicy(
                dimensions, alpha=0.70, seed=seed
            )
        ),
        "portfolio_linucb": lambda dimensions, seed: PortfolioAwareLinUCBPolicy(
            dimensions,
            alpha=0.70,
            base_shadow_price=0.22,
            seed=seed,
        ),
        "portfolio_thompson": (
            lambda dimensions, seed: PortfolioAwareThompsonPolicy(
                dimensions,
                posterior_scale=0.30,
                base_shadow_price=0.22,
                seed=seed,
            )
        ),
        "cmdp_fair_hard": lambda dimensions, seed: HardFairCMDPPolicy(
            base_policy=PortfolioAwareLinUCBPolicy(
                dimensions,
                alpha=0.70,
                base_shadow_price=0.22,
                seed=seed,
            ),
            target_shares=target_shares,
            parity_tolerance=parity_tolerance,
            seed=seed,
        ),
        "cmdp_fair_lagrangian": (
            lambda dimensions, seed: LagrangianFairCMDPPolicy(
                base_policy=PortfolioAwareLinUCBPolicy(
                    dimensions,
                    alpha=0.70,
                    base_shadow_price=0.22,
                    seed=seed,
                ),
                config=lagrangian_config,
                seed=seed,
            )
        ),
        "oracle_portfolio": lambda dimensions, seed: OraclePortfolioPolicy(
            dimensions,
            state_aware=True,
            seed=seed,
        ),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for running experiments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run Phase 1 arm bandits, Phase 2 contextual bandits, "
            "Phase 3 CMDP, or all phases"
        )
    )
    parser.add_argument(
        "--mode",
        choices=tuple(MODE_ALIASES),
        help="1/arm, 2/contextual, 3/cmdp, or 4/all; omit for an interactive menu",
    )
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=150)
    parser.add_argument(
        "--budget",
        type=float,
        help="one-shot budget in Phases 1/2; initial capital stock in Phase 3",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--arms-config", type=Path)
    parser.add_argument("--pool-size", type=int, default=20)
    parser.add_argument("--feedback-delay", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument(
        "--repayment-delay",
        "--delay",
        dest="repayment_delay",
        type=int,
        default=8,
    )
    parser.add_argument("--recycle-fraction", type=float, default=1.0)
    parser.add_argument("--default-recovery", type=float, default=0.0)
    parser.add_argument("--liquidity-reserve", type=float, default=0.0)
    parser.add_argument("--parity-tolerance", type=float, default=0.05)
    args = parser.parse_args(argv)
    if args.seeds <= 0:
        parser.error("--seeds must be positive")
    if args.rounds <= 0:
        parser.error("--rounds must be positive")
    if args.budget is not None and args.budget <= 0:
        parser.error("--budget must be positive")
    normalized_mode = MODE_ALIASES.get(args.mode) if args.mode else None
    if normalized_mode in {"arm", "cmdp"} and args.feedback_delay != 0:
        parser.error("--feedback-delay is only valid for contextual mode")
    if normalized_mode in {"contextual", "cmdp"} and args.arms_config is not None:
        parser.error("--arms-config is only valid for arm mode")
    if args.pool_size <= 0:
        parser.error("--pool-size must be positive")
    if args.feedback_delay < 0:
        parser.error("--feedback-delay cannot be negative")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.repayment_delay < 1:
        parser.error("--repayment-delay must be at least one")
    if not 0.0 <= args.recycle_fraction <= 1.0:
        parser.error("--recycle-fraction must be in [0, 1]")
    if not 0.0 <= args.default_recovery <= 1.0:
        parser.error("--default-recovery must be in [0, 1]")
    if args.liquidity_reserve < 0:
        parser.error("--liquidity-reserve cannot be negative")
    if not 0.0 <= args.parity_tolerance <= 1.0:
        parser.error("--parity-tolerance must be in [0, 1]")
    return args


def run_arm(
    args: argparse.Namespace,
    output: Path | None = None,
):
    """Run an arm bandit experiment with the specified arguments."""
    try:
        arms, targets = (
            load_arm_configuration(args.arms_config)
            if args.arms_config
            else default_arm_configuration()
        )
    except ValueError as exc:
        raise SystemExit(f"Invalid arm configuration: {exc}") from exc
    budget = args.budget if args.budget is not None else 600.0

    def environment_factory(seed: int) -> BanditEnvironment:
        return BanditEnvironment.for_arms(
            arms=arms,
            config=EnvironmentConfig(
                total_budget=budget,
                max_rounds=args.rounds,
                seed=seed,
            ),
        )

    summaries, trajectories = compare_policies(
        environment_factory,
        build_arm_policy_factories(targets),
        targets,
        list(range(args.seeds)),
    )
    destination = output or args.output or Path("outputs") / "arm"
    aggregate = save_results(summaries, trajectories, destination, "arm")
    display_results(aggregate, destination, "arm")
    return aggregate


def run_contextual(
    args: argparse.Namespace,
    output: Path | None = None,
):
    """Run a contextual bandit experiment with the specified arguments."""
    budget = args.budget if args.budget is not None else 600_000.0
    encoder = BorrowerFeatureEncoder(include_protected_attributes=False)
    targets = DEFAULT_TARGET_SHARES

    def environment_factory(seed: int) -> BanditEnvironment:
        return BanditEnvironment.for_contextual(
            config=ContextualEnvironmentConfig(
                total_budget=budget,
                max_rounds=args.rounds,
                seed=seed,
                pool_size=args.pool_size,
                feedback_delay_rounds=args.feedback_delay,
                minimum_cost=1.0,
            ),
            encoder=encoder,
        )

    summaries, trajectories = compare_policies(
        environment_factory,
        build_contextual_policy_factories(targets),
        targets,
        list(range(args.seeds)),
    )
    destination = output or args.output or Path("outputs") / "contextual"
    aggregate = save_results(
        summaries,
        trajectories,
        destination,
        "contextual",
    )
    display_results(aggregate, destination, "contextual")
    return aggregate


def run_cmdp(
    args: argparse.Namespace,
    output: Path | None = None,
):
    """Run Phase 3 with delayed feedback and recyclable capital."""

    capital = args.budget if args.budget is not None else 600_000.0
    encoder = BorrowerFeatureEncoder(include_protected_attributes=False)
    config = CMDPEnvironmentConfig(
        initial_capital=capital,
        max_rounds=args.rounds,
        pool_size=args.pool_size,
        max_loans_per_round=args.batch_size,
        repayment_delay_rounds=args.repayment_delay,
        recycle_fraction=args.recycle_fraction,
        default_recovery_fraction=args.default_recovery,
        minimum_liquidity_reserve=args.liquidity_reserve,
    )
    summaries, trajectories, funding_log, maturity_log = compare_cmdp_policies(
        env_config=config,
        policy_factories=build_cmdp_policy_factories(
            DEFAULT_TARGET_SHARES,
            args.parity_tolerance,
        ),
        target_shares=DEFAULT_TARGET_SHARES,
        parity_tolerance=args.parity_tolerance,
        seeds=list(range(args.seeds)),
        encoder=encoder,
    )
    destination = output or args.output or Path("outputs") / "cmdp"
    aggregate = save_cmdp_results(
        summaries,
        trajectories,
        funding_log,
        maturity_log,
        destination,
    )
    display_results(aggregate, destination, "cmdp")
    return aggregate


def combine_aggregate_results(
    results: Mapping[str, "pd.DataFrame"],
):
    """Flatten each phase's aggregate metrics into one comparable report."""

    import pandas as pd

    frames: list[pd.DataFrame] = []
    for mode, aggregate in results.items():
        flattened = aggregate.copy()
        flattened.columns = [
            "_".join(str(part) for part in column if part)
            for column in flattened.columns
        ]
        flattened = flattened.reset_index()
        flattened.insert(0, "mode", mode)
        frames.append(flattened)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    if "cumulative_reward_mean" in combined:
        combined["rank_within_mode"] = combined.groupby("mode")[
            "cumulative_reward_mean"
        ].rank(method="min", ascending=False)
    return combined


def run_all(args: argparse.Namespace):
    """Run Phases 1, 2, and 3 and write a combined performance table."""

    output = args.output or Path("outputs") / "all"
    results = {
        "arm": run_arm(args, output / "arm"),
        "contextual": run_contextual(args, output / "contextual"),
        "cmdp": run_cmdp(args, output / "cmdp"),
    }
    combined = combine_aggregate_results(results)
    output.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output / "combined_aggregate_results.csv", index=False)
    print("\nAll phases combined metrics:\n")
    display_columns = [
        column
        for column in (
            "mode",
            "policy",
            "rank_within_mode",
            "cumulative_reward_mean",
            "repayment_rate_mean",
            "reward_per_cost_mean",
            "budget_utilization_mean",
            "capital_turnover_mean",
            "capital_survival_ratio_mean",
            "demographic_parity_gap_mean",
            "quota_satisfaction_rate_mean",
        )
        if column in combined
    ]
    print(combined[display_columns].round(5).to_string(index=False))
    print(f"\nSaved combined results to: {output.resolve()}")
    return combined


def prompt_for_mode() -> str:
    print("\nChoose an experiment:")
    print("  1. Arm bandits on simulated mean/std borrower groups")
    print("  2. Contextual bandits on simulated borrower applications")
    print("  3. Stateful contextual CMDP")
    print("  4. Run all phases and compare metrics")
    try:
        choice = input("Selection [1-4]: ").strip().lower()
    except EOFError as exc:
        raise SystemExit("No mode selected; pass --mode 1, 2, 3, or 4") from exc
    if choice not in MODE_ALIASES:
        raise SystemExit("Invalid selection; choose 1, 2, 3, or 4")
    return MODE_ALIASES[choice]


def main() -> None:
    args = parse_args()
    mode = MODE_ALIASES[args.mode] if args.mode else prompt_for_mode()
    if mode == "arm":
        run_arm(args)
    elif mode == "contextual":
        run_contextual(args)
    elif mode == "cmdp":
        run_cmdp(args)
    else:
        run_all(args)


if __name__ == "__main__":
    main()
