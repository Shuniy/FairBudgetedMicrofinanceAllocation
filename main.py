from __future__ import annotations

import argparse
import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from pathlib import Path

from Algorithms.bandit import RandomPolicy
from Algorithms.budgeted_ucb import BudgetedUCBPolicy
from Algorithms.cmdp import (
    ContextOnlyBudgetedLinUCBPolicy,
    HardFairCMDPPolicy,
    LagrangianFairCMDPPolicy,
    OraclePortfolioPolicy,
    PortfolioAwareEpsilonGreedyPolicy,
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
    ARM_PARITY_TOLERANCE,
    CONTEXTUAL_PARITY_TOLERANCE,
    QUOTA_TOLERANCE,
    compare_cmdp_policies,
    compare_policies,
    display_results,
    episode_completion,
    flatten_aggregate,
    rank_results,
    reproducibility_metadata,
    save_combined_report,
    save_cmdp_results,
    save_evaluation_report,
    save_results,
    winner_summary,
)
from models.environment_config import (
    CMDPEnvironmentConfig,
    ContextualEnvironmentConfig,
    EnvironmentConfig,
)
from models.fairness_config import LagrangianConfig
from robustness import (
    BUDGET_SWEEP_VALUES,
    FAIRNESS_SWEEP_VALUES,
    IMBALANCE_SWEEP_VALUES,
    QUICK_BUDGET_SWEEP_VALUES,
    QUICK_FAIRNESS_SWEEP_VALUES,
    QUICK_IMBALANCE_SWEEP_VALUES,
    CMDPSweepCase,
    run_cmdp_sweep,
    save_mode4_report,
    validate_mode4_baselines,
)


ROOT = Path(__file__).resolve().parent
MASTER_SEED = 5726455
OPTIONS = dict(
    seeds=[10, 12, 15], rounds=[120, 150, 180, 210, 240],
    arm_budget=[420, 500, 600, 780, 900, 1000],
    budget=[420_000, 500_000, 600_000, 780_000, 900_000, 1_000_000],
    pool_size=[16, 20, 24, 28, 32], feedback_delay=[2, 3, 4],
    batch_size=[2, 3, 4, 5], repayment_delay=[5, 8, 11, 14],
    recycle_fraction=[0.70, 0.80, 0.85, 0.90, 1.00],
    default_recovery=[0.00, 0.10, 0.15, 0.20, 0.25],
    liquidity_reserve=[10_000, 20_000, 30_000, 40_000],
)


def generate_scenarios(num_scenarios=10, master_seed=MASTER_SEED):
    """Generate named, repeatable scenarios from the parameter options."""
    if not 1 <= num_scenarios <= 10000:
        raise ValueError("num_scenarios must be between 1 and 10000")
    rng = random.Random(master_seed)
    scenarios, used_blocks, used_configs = {}, set(), set()
    for index in range(num_scenarios):
        block = rng.randrange(100, 1_000_000)
        while block in used_blocks:
            block = rng.randrange(100, 1_000_000)
        used_blocks.add(block)
        cfg = {key: rng.choice(values) for key, values in OPTIONS.items()}
        identity = tuple(cfg[key] for key in OPTIONS if key != "seeds")
        while identity in used_configs:
            cfg = {key: rng.choice(values) for key, values in OPTIONS.items()}
            identity = tuple(cfg[key] for key in OPTIONS if key != "seeds")
        used_configs.add(identity)
        scenarios[f"scenario_{index + 1}"] = dict(seed_start=100 * block + 1, **cfg)
    return scenarios


# Generate ten scenarios from the parameter choices above.
SCENARIOS = generate_scenarios(num_scenarios=10, master_seed=MASTER_SEED)


MODE_ALIASES = {
    "1": "arm",
    "2": "contextual",
    "3": "cmdp",
    "4": "sweeps",
    "5": "all",
    "arm": "arm",
    "contextual": "contextual",
    "cmdp": "cmdp",
    "all": "all",
    "sweeps": "sweeps",
    "robustness": "sweeps",
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
            parity_tolerance=ARM_PARITY_TOLERANCE,
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
        "fair_epsilon_greedy_hard": lambda env, seed: HardFairPolicy(
            ContextualEpsilonGreedyPolicy(
                dimensions(env), epsilon=0.10, budget_aware=True, seed=seed
            ),
            target_shares,
            parity_tolerance=CONTEXTUAL_PARITY_TOLERANCE,
            seed=seed,
        ),
        "fair_epsilon_greedy_soft": lambda env, seed: SoftFairPolicy(
            ContextualEpsilonGreedyPolicy(
                dimensions(env), epsilon=0.10, budget_aware=True, seed=seed
            ),
            target_shares,
            penalty_lambda=0.015,
            seed=seed,
        ),
        "fair_linucb_hard": lambda env, seed: HardFairPolicy(
            LinUCBPolicy(
                dimensions(env), alpha=0.70, budget_aware=True, seed=seed
            ),
            target_shares,
            parity_tolerance=CONTEXTUAL_PARITY_TOLERANCE,
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
        "fair_thompson_hard": lambda env, seed: HardFairPolicy(
            ContextualThompsonSamplingPolicy(
                dimensions(env), posterior_scale=0.30, budget_aware=True, seed=seed
            ),
            target_shares,
            parity_tolerance=CONTEXTUAL_PARITY_TOLERANCE,
            seed=seed,
        ),
        "fair_thompson_soft": lambda env, seed: SoftFairPolicy(
            ContextualThompsonSamplingPolicy(
                dimensions(env), posterior_scale=0.30, budget_aware=True, seed=seed
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
    *,
    lagrangian_strength_scale: float = 1.0,
):
    """Build Phase 3 policies while reusing the parent contextual ridge model."""

    lagrangian_config = LagrangianConfig(
        target_shares=target_shares,
        parity_tolerance=parity_tolerance,
        quota_tolerance=QUOTA_TOLERANCE,
        dual_learning_rate=0.20,
        initial_lambda_quota=0.20,
        initial_lambda_parity=0.20,
        max_lambda=8.0,
        warmup_rounds=5,
        strength_scale=lagrangian_strength_scale,
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
        "portfolio_epsilon_greedy": (
            lambda dimensions, seed: PortfolioAwareEpsilonGreedyPolicy(
                dimensions,
                epsilon=0.10,
                base_shadow_price=0.22,
                seed=seed,
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
        "cmdp_fair_hard_epsilon_greedy": lambda dimensions, seed: HardFairCMDPPolicy(
            base_policy=PortfolioAwareEpsilonGreedyPolicy(
                dimensions,
                epsilon=0.10,
                base_shadow_price=0.22,
                seed=seed,
            ),
            target_shares=target_shares,
            parity_tolerance=parity_tolerance,
            seed=seed,
        ),
        "cmdp_fair_lagrangian_epsilon_greedy": (
            lambda dimensions, seed: LagrangianFairCMDPPolicy(
                base_policy=PortfolioAwareEpsilonGreedyPolicy(
                    dimensions,
                    epsilon=0.10,
                    base_shadow_price=0.22,
                    seed=seed,
                ),
                config=lagrangian_config,
                seed=seed,
            )
        ),
        "cmdp_fair_hard_linucb": lambda dimensions, seed: HardFairCMDPPolicy(
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
        "cmdp_fair_lagrangian_linucb": (
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
        "cmdp_fair_hard_thompson": lambda dimensions, seed: HardFairCMDPPolicy(
            base_policy=PortfolioAwareThompsonPolicy(
                dimensions,
                posterior_scale=0.30,
                base_shadow_price=0.22,
                seed=seed,
            ),
            target_shares=target_shares,
            parity_tolerance=parity_tolerance,
            seed=seed,
        ),
        "cmdp_fair_lagrangian_thompson": (
            lambda dimensions, seed: LagrangianFairCMDPPolicy(
                base_policy=PortfolioAwareThompsonPolicy(
                    dimensions,
                    posterior_scale=0.30,
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
            "Phase 3 CMDP, all phases, or Phase 3 robustness sweeps"
        )
    )
    parser.add_argument(
        "--mode",
        choices=tuple(MODE_ALIASES),
        help=(
            "1/arm, 2/contextual, 3/cmdp, 4/sweeps, or 5/all; "
            "omit for an interactive menu"
        ),
    )
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
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
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help=(
            "Modes 4/5: use two seeds and two values for each robustness sweep"
        ),
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Mode 5: independently verify saved results after each stage",
    )
    args = parser.parse_args(argv)
    if args.seeds <= 0:
        parser.error("--seeds must be positive")
    if args.rounds <= 0:
        parser.error("--rounds must be positive")
    if args.budget is not None and args.budget <= 0:
        parser.error("--budget must be positive")
    normalized_mode = MODE_ALIASES.get(args.mode) if args.mode else None
    if normalized_mode in {"arm", "cmdp", "sweeps"} and args.feedback_delay != 0:
        parser.error("--feedback-delay is only valid for contextual mode")
    if normalized_mode in {"contextual", "cmdp", "sweeps"} and args.arms_config is not None:
        parser.error("--arms-config is only valid for arm mode")
    if args.pool_size <= 0:
        parser.error("--pool-size must be positive")
    if args.feedback_delay < 0:
        parser.error("--feedback-delay cannot be negative")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.repayment_delay < 0:
        parser.error("--repayment-delay cannot be negative")
    if not 0.0 <= args.recycle_fraction <= 1.0:
        parser.error("--recycle-fraction must be in [0, 1]")
    if not 0.0 <= args.default_recovery <= 1.0:
        parser.error("--default-recovery must be in [0, 1]")
    if args.liquidity_reserve < 0:
        parser.error("--liquidity-reserve cannot be negative")
    if not 0.0 <= args.parity_tolerance <= 1.0:
        parser.error("--parity-tolerance must be in [0, 1]")
    if args.quick_test and normalized_mode not in {None, "sweeps", "all"}:
        parser.error("--quick-test is only valid for Modes 4 and 5")
    if args.verify and normalized_mode not in {None, "all"}:
        parser.error("--verify is only valid for Mode 5")
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
    config = EnvironmentConfig(total_budget=budget, max_rounds=args.rounds)
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    policies = build_arm_policy_factories(targets)

    def environment_factory(seed: int) -> BanditEnvironment:
        return BanditEnvironment.for_arms(
            arms=arms,
            config=replace(config, seed=seed),
        )

    summaries, trajectories = compare_policies(
        environment_factory,
        policies,
        targets,
        seeds,
    )
    destination = output or args.output or Path("outputs") / "arm"
    aggregate = save_results(summaries, trajectories, destination, "arm")
    manifest = {
        "mode": "arm",
        "seeds": seeds,
        "policies": list(policies),
        "configuration": {
            **asdict(config),
            "arms": [asdict(arm) for arm in arms],
            "target_shares": targets,
        },
        "fairness_thresholds": {
            "parity_tolerance": ARM_PARITY_TOLERANCE,
            "quota_tolerance": QUOTA_TOLERANCE,
        },
        "policy_configurations": summaries.attrs["policy_configurations"],
        **episode_completion(len(seeds) * len(policies), len(summaries)),
        **reproducibility_metadata(),
    }
    (destination / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
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
    config = ContextualEnvironmentConfig(
        total_budget=budget,
        max_rounds=args.rounds,
        pool_size=args.pool_size,
        feedback_delay_rounds=args.feedback_delay,
        minimum_cost=1.0,
    )

    def environment_factory(seed: int) -> BanditEnvironment:
        return BanditEnvironment.for_contextual(
            config=replace(config, seed=seed),
            encoder=encoder,
        )

    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    summaries, trajectories = compare_policies(
        environment_factory,
        build_contextual_policy_factories(targets),
        targets,
        seeds,
    )
    destination = output or args.output or Path("outputs") / "contextual"
    aggregate = save_results(
        summaries,
        trajectories,
        destination,
        "contextual",
    )
    ranked = save_evaluation_report(
        aggregate=aggregate,
        output=destination,
        mode="contextual",
        parity_tolerance=CONTEXTUAL_PARITY_TOLERANCE,
        quota_tolerance=QUOTA_TOLERANCE,
        seeds=seeds,
        configuration={
            **asdict(config),
            "target_shares": targets,
            "protected_attributes_in_prediction": False,
        },
        policy_parameters={
            "epsilon": 0.10,
            "linucb_alpha": 0.70,
            "thompson_posterior_scale": 0.30,
            "hard_parity_tolerance": CONTEXTUAL_PARITY_TOLERANCE,
            "soft_penalty_lambda": 0.015,
        },
        policy_configurations=summaries.attrs["policy_configurations"],
    )
    display_results(aggregate, destination, "contextual", ranked)
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
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    summaries, trajectories, funding_log, maturity_log = compare_cmdp_policies(
        env_config=config,
        policy_factories=build_cmdp_policy_factories(
            DEFAULT_TARGET_SHARES,
            args.parity_tolerance,
        ),
        target_shares=DEFAULT_TARGET_SHARES,
        parity_tolerance=args.parity_tolerance,
        seeds=seeds,
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
    ranked = save_evaluation_report(
        aggregate=aggregate,
        output=destination,
        mode="cmdp",
        parity_tolerance=args.parity_tolerance,
        quota_tolerance=QUOTA_TOLERANCE,
        seeds=seeds,
        configuration={
            **asdict(config),
            "target_shares": DEFAULT_TARGET_SHARES,
            "protected_attributes_in_prediction": False,
        },
        policy_parameters={
            "epsilon": 0.10,
            "linucb_alpha": 0.70,
            "thompson_posterior_scale": 0.30,
            "base_shadow_price": 0.22,
            "lagrangian": {
                "quota_tolerance": QUOTA_TOLERANCE,
                "dual_learning_rate": 0.20,
                "initial_lambda_quota": 0.20,
                "initial_lambda_parity": 0.20,
                "max_lambda": 8.0,
                "warmup_rounds": 5,
            },
        },
        policy_configurations=summaries.attrs["policy_configurations"],
    )
    display_results(aggregate, destination, "cmdp", ranked)
    return aggregate


def combine_aggregate_results(
    results: Mapping[str, "pd.DataFrame"],
):
    """Combine phase results with separate columns for phase-specific parity."""

    import pandas as pd

    frames: list[pd.DataFrame] = []
    for mode, aggregate in results.items():
        flattened = flatten_aggregate(
            aggregate.drop(columns="demographic_parity_gap", level=0, errors="ignore")
        )
        flattened.insert(0, "mode", mode)
        frames.append(flattened)
    combined = pd.concat(frames, ignore_index=True, sort=False).copy()
    if "cumulative_reward_mean" in combined:
        combined["rank_within_mode"] = combined.groupby("mode")[
            "cumulative_reward_mean"
        ].rank(method="min", ascending=False)
    return combined


def run_all(args: argparse.Namespace):
    """Run Modes 1–4 using their shared output directories."""

    output = args.output or Path("outputs")
    suite_args = args
    if args.quick_test:
        suite_args = argparse.Namespace(**vars(args))
        suite_args.seeds = min(args.seeds, 2)
    results = {
        "arm": run_arm(suite_args, output / "arm"),
        "contextual": run_contextual(suite_args, output / "contextual"),
        "cmdp": run_cmdp(suite_args, output / "cmdp"),
    }
    combined = combine_aggregate_results(results)
    output.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output / "combined_aggregate_results.csv", index=False)
    print("\nModes 1–3 combined metrics:\n")
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
            "allocation_share_gap_mean",
            "feasible_dp_gap_mean",
            "population_dp_gap_mean",
            "quota_satisfaction_rate_mean",
        )
        if column in combined
    ]
    print(combined[display_columns].round(5).to_string(index=False))
    ranked_results = {
        "contextual": rank_results(
            results["contextual"],
            CONTEXTUAL_PARITY_TOLERANCE,
            QUOTA_TOLERANCE,
        ),
        "cmdp": rank_results(
            results["cmdp"],
            suite_args.parity_tolerance,
            QUOTA_TOLERANCE,
        ),
    }
    if args.verify:
        verify_suite_stage(output, "verification_defaults.json")
    run_cmdp_sweeps(suite_args, output / "cmdp_robustness")
    if args.verify:
        verify_suite_stage(output, "verification_robustness.json")
    suite_manifest = {
        "mode": 5,
        "name": "complete_experiment_suite",
        "included_modes": {
            "1": "arm",
            "2": "contextual",
            "3": "cmdp",
            "4": "cmdp_robustness_sweeps",
        },
        "quick_test": bool(args.quick_test),
        "arguments": vars(suite_args),
        "seeds": list(range(suite_args.seed_start, suite_args.seed_start + suite_args.seeds)),
        **reproducibility_metadata(),
        "outputs": {
            "mode_1": "arm",
            "mode_2": "contextual",
            "mode_3": "cmdp",
            "mode_4": "cmdp_robustness",
            "combined_aggregates": "combined_aggregate_results.csv",
            "combined_report": "report.html",
            "cross_configuration": "all_scenarios",
        },
    }
    save_combined_report(
        ranked_results,
        output,
        arm_aggregate=results["arm"],
        sweep_report=output / "cmdp_robustness" / "report.html",
    )
    print("\nPhase winners (not compared across phases):\n")
    print(winner_summary(ranked_results["contextual"], "contextual"))
    print(winner_summary(ranked_results["cmdp"], "cmdp"))
    print(f"\nSaved combined results to: {output.resolve()}")
    # Run cross-configuration validation beneath the selected suite output.
    run_all_scenarios(output / "all_scenarios")
    directories = [output, *(output / "all_scenarios" / name for name in SCENARIOS)]
    children = [
        json.loads(path.read_text())
        for directory in directories
        for label in ("arm", "contextual", "cmdp", "cmdp_robustness")
        if (path := directory / label / "run_manifest.json").is_file()
    ]
    suite_manifest.update(episode_completion(
        sum(child["expected_episodes"] for child in children),
        sum(child["actual_episodes"] for child in children),
    ))
    if len(children) != 4 * len(directories) or any(child["status"] != "complete" for child in children):
        suite_manifest["status"] = "incomplete"
    suite_manifest["scenarios"] = SCENARIOS
    (output / "run_manifest.json").write_text(
        json.dumps(suite_manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    if args.verify:
        verify_suite_stage(output, "verification_summary.json")
    return combined


def verify_suite_stage(output: Path, filename: str) -> None:
    from verify_results import audit_outputs

    report = audit_outputs(output, summary_path=output / filename)
    print(f"Verified {report['episodes_checked']} episodes. {report['failed_checks']} failed checks.")
    if report["failed_checks"]:
        raise RuntimeError(f"Saved-output verification failed. See {output / filename}.")



def _only_cmdp_fairness_family(
    factories: Mapping[str, object],
    family: str,
):
    marker = f"cmdp_fair_{family}_"
    return {
        name: factory
        for name, factory in factories.items()
        if name.startswith(marker)
    }


def run_cmdp_sweeps(
    args: argparse.Namespace,
    output: Path | None = None,
):
    """Run Mode 4 robustness experiments using only Phase 3 CMDP policies."""

    capital = args.budget if args.budget is not None else 600_000.0
    base_config = CMDPEnvironmentConfig(
        initial_capital=capital,
        max_rounds=args.rounds,
        pool_size=args.pool_size,
        max_loans_per_round=args.batch_size,
        repayment_delay_rounds=args.repayment_delay,
        recycle_fraction=args.recycle_fraction,
        default_recovery_fraction=args.default_recovery,
        minimum_liquidity_reserve=args.liquidity_reserve,
    )
    seed_count = min(args.seeds, 2) if args.quick_test else args.seeds
    seeds = list(range(args.seed_start, args.seed_start + seed_count))
    budget_values = (
        QUICK_BUDGET_SWEEP_VALUES
        if args.quick_test
        else BUDGET_SWEEP_VALUES
    )
    imbalance_values = (
        QUICK_IMBALANCE_SWEEP_VALUES
        if args.quick_test
        else IMBALANCE_SWEEP_VALUES
    )
    fairness_values = (
        QUICK_FAIRNESS_SWEEP_VALUES
        if args.quick_test
        else FAIRNESS_SWEEP_VALUES
    )
    destination = output or args.output or Path("outputs") / "cmdp_robustness"
    sweeps_output = destination / "sweeps"
    encoder = BorrowerFeatureEncoder(include_protected_attributes=False)

    normal_factories = build_cmdp_policy_factories(
        DEFAULT_TARGET_SHARES,
        args.parity_tolerance,
    )
    budget_cases = [
        CMDPSweepCase(
            value=multiplier,
            env_config=replace(
                base_config,
                initial_capital=capital * multiplier,
            ),
            policy_factories=normal_factories,
            target_shares=DEFAULT_TARGET_SHARES,
            parity_tolerance=args.parity_tolerance,
            parameter_kind="initial_capital_multiplier",
            parameter_details={
                "budget_multiplier": multiplier,
                "configured_initial_capital": capital * multiplier,
            },
            is_baseline=bool(multiplier == 1.0),
        )
        for multiplier in budget_values
    ]
    print("\nMode 4A: running Phase 3 budget sensitivity sweep...")
    budget_artifact = run_cmdp_sweep(
        "budget",
        budget_cases,
        seeds,
        sweeps_output / "budget",
        base_config,
        QUOTA_TOLERANCE,
        baseline_value=1.0,
        encoder=encoder,
    )

    imbalance_cases = [
        CMDPSweepCase(
            value=minority_share,
            env_config=replace(
                base_config,
                marginalized_probability=minority_share,
            ),
            policy_factories=normal_factories,
            target_shares=DEFAULT_TARGET_SHARES,
            parity_tolerance=args.parity_tolerance,
            parameter_kind="marginalized_community_probability",
            parameter_details={
                "minority_share": minority_share,
                "minority_share_definition": (
                    "P(marginalized_community=True); the audited primary "
                    "minority label excludes overlapping disability/reserved "
                    "applicants because those labels have higher precedence"
                ),
            },
            is_baseline=bool(minority_share == base_config.marginalized_probability),
        )
        for minority_share in imbalance_values
    ]
    print("Mode 4B: running Phase 3 group-imbalance sweep...")
    imbalance_artifact = run_cmdp_sweep(
        "group_imbalance",
        imbalance_cases,
        seeds,
        sweeps_output / "group_imbalance",
        base_config,
        QUOTA_TOLERANCE,
        baseline_value=base_config.marginalized_probability,
        encoder=encoder,
    )

    fairness_cases: list[CMDPSweepCase] = []
    for strength in fairness_values:
        scaled_targets = {
            group: share * strength
            for group, share in DEFAULT_TARGET_SHARES.items()
        }
        hard_factories = _only_cmdp_fairness_family(
            build_cmdp_policy_factories(
                scaled_targets,
                args.parity_tolerance,
            ),
            "hard",
        )
        fairness_cases.append(
            CMDPSweepCase(
                value=strength,
                env_config=base_config,
                policy_factories=hard_factories,
                target_shares=scaled_targets,
                parity_tolerance=args.parity_tolerance,
                parameter_kind="hard_quota_target_scale",
                parameter_details={
                    "fairness_strength": strength,
                    "quota_scale": strength,
                    "parity_tolerance_held_fixed": args.parity_tolerance,
                },
                is_baseline=bool(strength == 1.0),
            )
        )
        lagrangian_factories = _only_cmdp_fairness_family(
            build_cmdp_policy_factories(
                DEFAULT_TARGET_SHARES,
                args.parity_tolerance,
                lagrangian_strength_scale=strength,
            ),
            "lagrangian",
        )
        fairness_cases.append(
            CMDPSweepCase(
                value=strength,
                env_config=base_config,
                policy_factories=lagrangian_factories,
                target_shares=DEFAULT_TARGET_SHARES,
                parity_tolerance=args.parity_tolerance,
                parameter_kind="lagrangian_penalty_scale",
                parameter_details={
                    "fairness_strength": strength,
                    "lagrangian_strength_scale": strength,
                    "base_initial_lambda_quota": 0.20,
                    "base_initial_lambda_parity": 0.20,
                    "dual_learning_rate_held_fixed": 0.20,
                },
                is_baseline=bool(strength == 1.0),
            )
        )
    print("Mode 4C: running Phase 3 fairness-strength sweep...")
    fairness_artifact = run_cmdp_sweep(
        "fairness_strength",
        fairness_cases,
        seeds,
        sweeps_output / "fairness_strength",
        base_config,
        QUOTA_TOLERANCE,
        baseline_value=1.0,
        encoder=encoder,
    )

    artifacts = {
        "budget": budget_artifact,
        "group_imbalance": imbalance_artifact,
        "fairness_strength": fairness_artifact,
    }
    baseline_consistency = validate_mode4_baselines(artifacts, destination)
    report = save_mode4_report(
        artifacts,
        destination,
        seeds,
        args.quick_test,
        baseline_consistency,
    )
    episodes = sum(len(artifact.raw_results) for artifact in artifacts.values())
    print(f"\nMode 4 completed {episodes} Phase 3 CMDP episodes.")
    print(f"Saved robustness outputs to: {destination.resolve()}")
    print(f"HTML report: {report.resolve()}")
    return artifacts


def run_scenario(name: str, output_root: Path = ROOT / "outputs") -> None:
    """Run Modes 1–4 for one generated scenario."""
    config = SCENARIOS[name]
    destination = Path(output_root) / name
    for mode, label, experiment in (
        ("1", "arm", run_arm),
        ("2", "contextual", run_contextual),
        ("3", "cmdp", run_cmdp),
        ("4", "cmdp_robustness", run_cmdp_sweeps),
    ):
        options = {key: value for key, value in config.items() if key != "arm_budget"}
        if mode == "1":
            options["budget"] = config["arm_budget"]
        if mode != "2":
            options["feedback_delay"] = 0
        argv = ["--mode", mode]
        for key, value in options.items():
            argv.extend(["--" + key.replace("_", "-"), str(value)])
        args = parse_args(argv)
        phase_output = destination / label
        phase_output.mkdir(parents=True, exist_ok=True)
        print(f"Running {name}: {label} -> {phase_output.resolve()}", flush=True)
        experiment(args, phase_output)
        print(f"Completed {name}: {label}", flush=True)


def run_all_scenarios(
    output_root: Path = ROOT / "outputs" / "all_scenarios",
) -> None:
    """Run every generated scenario, creating its output folder up front."""
    scenarios = list(SCENARIOS)
    output_root = Path(output_root).resolve()
    # Make every selected scenario visible before the first long experiment.
    for scenario in scenarios:
        (output_root / scenario).mkdir(parents=True, exist_ok=True)
    print(
        f"Running {len(scenarios)} scenario(s) sequentially in {output_root}: "
        f"{', '.join(scenarios)}",
        flush=True,
    )
    for index, scenario in enumerate(scenarios, start=1):
        print(f"Scenario {index}/{len(scenarios)}: {scenario}", flush=True)
        run_scenario(scenario, output_root)


def prompt_for_mode() -> str:
    print("\nChoose an experiment:")
    print("  1. Arm bandits on simulated mean/std borrower groups")
    print("  2. Contextual bandits on simulated borrower applications")
    print("  3. Stateful contextual CMDP")
    print("  4. Phase 3 CMDP robustness sweeps")
    print("  5. Run Modes 1–4 with cross configuration validation and collect all reports")
    try:
        choice = input("Selection [1-5]: ").strip().lower()
    except EOFError as exc:
        raise SystemExit("No mode selected; pass --mode 1, 2, 3, 4, or 5") from exc
    if choice not in MODE_ALIASES:
        raise SystemExit("Invalid selection; choose 1, 2, 3, 4, or 5")
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
    elif mode == "all":
        run_all(args)
    else:
        run_cmdp_sweeps(args)


if __name__ == "__main__":
    main()
