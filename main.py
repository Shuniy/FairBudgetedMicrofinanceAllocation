from __future__ import annotations

import argparse
from pathlib import Path

from Algorithms.bandit import RandomPolicy
from Algorithms.budgeted_ucb import BudgetedUCBPolicy
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
from evaluate import compare_policies, display_results, save_results
from models.environment_config import ContextualEnvironmentConfig, EnvironmentConfig


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


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for running experiments."""
    parser = argparse.ArgumentParser(
        description="Run fair budgeted arm or contextual bandit experiments"
    )
    parser.add_argument("--mode", choices=("arm", "contextual"), required=True)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=150)
    parser.add_argument("--budget", type=float)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--arms-config", type=Path)
    parser.add_argument("--pool-size", type=int, default=20)
    parser.add_argument("--feedback-delay", type=int, default=0)
    args = parser.parse_args()
    if args.seeds <= 0:
        parser.error("--seeds must be positive")
    if args.rounds <= 0:
        parser.error("--rounds must be positive")
    if args.budget is not None and args.budget <= 0:
        parser.error("--budget must be positive")
    if args.mode == "arm" and args.feedback_delay != 0:
        parser.error("--feedback-delay is only valid for contextual mode")
    if args.mode == "contextual" and args.arms_config is not None:
        parser.error("--arms-config is only valid for arm mode")
    if args.mode == "contextual" and args.pool_size <= 0:
        parser.error("--pool-size must be positive")
    return args


def run_arm(args: argparse.Namespace) -> None:
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
    output = args.output or Path("outputs") / "arm"
    aggregate = save_results(summaries, trajectories, output, "arm")
    display_results(aggregate, output, "arm")


def run_contextual(args: argparse.Namespace) -> None:
    """Run a contextual bandit experiment with the specified arguments."""
    budget = args.budget if args.budget is not None else 600_000.0
    encoder = BorrowerFeatureEncoder(include_protected_attributes=False)
    targets = {"disabled": 0.04, "reserved": 0.08, "minority": 0.12, "women": 0.25}

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
    output = args.output or Path("outputs") / "contextual"
    aggregate = save_results(summaries, trajectories, output, "contextual")
    display_results(aggregate, output, "contextual")


def main() -> None:
    args = parse_args()
    if args.mode == "arm":
        run_arm(args)
    else:
        run_contextual(args)


if __name__ == "__main__":
    main()
