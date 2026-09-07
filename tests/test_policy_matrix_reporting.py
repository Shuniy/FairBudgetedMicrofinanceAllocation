from __future__ import annotations

import json

import numpy as np
import pandas as pd

from Algorithms.cmdp import (
    HardFairCMDPPolicy,
    LagrangianFairCMDPPolicy,
    PortfolioAwareEpsilonGreedyPolicy,
    PortfolioAwareThompsonPolicy,
)
from Algorithms.contextual import (
    ContextualEpsilonGreedyPolicy,
    ContextualThompsonSamplingPolicy,
)
from Algorithms.fairness import HardFairPolicy, SoftFairPolicy
from Data.feature_encoder import BorrowerFeatureEncoder
from Environments.base_environment import BanditEnvironment
from Environments.cmdp_environment import StatefulMicrofinanceCMDPEnv
from evaluate import (
    compare_cmdp_policies,
    compare_policies,
    format_ranked_table,
    rank_results,
    save_combined_report,
    save_evaluation_report,
    winner_summary,
)
from main import (
    DEFAULT_TARGET_SHARES,
    build_cmdp_policy_factories,
    build_contextual_policy_factories,
)
from models.environment_config import CMDPEnvironmentConfig, ContextualEnvironmentConfig


def contextual_environment(seed: int = 0) -> BanditEnvironment:
    return BanditEnvironment.for_contextual(
        ContextualEnvironmentConfig(
            total_budget=100_000,
            max_rounds=3,
            pool_size=5,
            seed=seed,
        )
    )


def test_contextual_policy_matrix_has_all_fourteen_policies() -> None:
    factories = build_contextual_policy_factories(DEFAULT_TARGET_SHARES)
    assert len(factories) == 14
    env = contextual_environment()

    hard_epsilon = factories["fair_epsilon_greedy_hard"](env, 0)
    soft_epsilon = factories["fair_epsilon_greedy_soft"](env, 0)
    hard_thompson = factories["fair_thompson_hard"](env, 0)
    soft_thompson = factories["fair_thompson_soft"](env, 0)
    assert isinstance(hard_epsilon, HardFairPolicy)
    assert isinstance(hard_epsilon.base_policy, ContextualEpsilonGreedyPolicy)
    assert isinstance(soft_epsilon, SoftFairPolicy)
    assert isinstance(soft_epsilon.base_policy, ContextualEpsilonGreedyPolicy)
    assert isinstance(hard_thompson, HardFairPolicy)
    assert isinstance(hard_thompson.base_policy, ContextualThompsonSamplingPolicy)
    assert isinstance(soft_thompson, SoftFairPolicy)
    assert isinstance(soft_thompson.base_policy, ContextualThompsonSamplingPolicy)


def test_cmdp_policy_matrix_has_all_twelve_policies() -> None:
    factories = build_cmdp_policy_factories(DEFAULT_TARGET_SHARES, 0.05)
    assert len(factories) == 12
    hard_epsilon = factories["cmdp_fair_hard_epsilon_greedy"](8, 0)
    lagrangian_epsilon = factories[
        "cmdp_fair_lagrangian_epsilon_greedy"
    ](8, 0)
    hard_thompson = factories["cmdp_fair_hard_thompson"](8, 0)
    lagrangian_thompson = factories[
        "cmdp_fair_lagrangian_thompson"
    ](8, 0)
    assert isinstance(hard_epsilon, HardFairCMDPPolicy)
    assert isinstance(hard_epsilon.base_policy, PortfolioAwareEpsilonGreedyPolicy)
    assert isinstance(lagrangian_epsilon, LagrangianFairCMDPPolicy)
    assert isinstance(
        lagrangian_epsilon.base_policy, PortfolioAwareEpsilonGreedyPolicy
    )
    assert isinstance(hard_thompson.base_policy, PortfolioAwareThompsonPolicy)
    assert isinstance(lagrangian_thompson.base_policy, PortfolioAwareThompsonPolicy)


def test_every_combination_runs_and_repeats_identically() -> None:
    contextual_factories = build_contextual_policy_factories(DEFAULT_TARGET_SHARES)

    def environment_factory(seed: int) -> BanditEnvironment:
        return contextual_environment(seed)

    first_summaries, _ = compare_policies(
        environment_factory,
        contextual_factories,
        DEFAULT_TARGET_SHARES,
        [0],
    )
    second_summaries, _ = compare_policies(
        environment_factory,
        contextual_factories,
        DEFAULT_TARGET_SHARES,
        [0],
    )
    assert len(first_summaries) == 14
    pd.testing.assert_frame_equal(first_summaries, second_summaries)

    encoder = BorrowerFeatureEncoder(include_protected_attributes=False)
    config = CMDPEnvironmentConfig(
        initial_capital=120_000,
        max_rounds=3,
        pool_size=5,
        max_loans_per_round=1,
        repayment_delay_rounds=2,
    )
    cmdp_factories = build_cmdp_policy_factories(DEFAULT_TARGET_SHARES, 0.05)
    first_cmdp, first_trajectories, _, _ = compare_cmdp_policies(
        config,
        cmdp_factories,
        DEFAULT_TARGET_SHARES,
        0.05,
        [0],
        encoder,
    )
    second_cmdp, second_trajectories, _, _ = compare_cmdp_policies(
        config,
        cmdp_factories,
        DEFAULT_TARGET_SHARES,
        0.05,
        [0],
        encoder,
    )
    assert len(first_cmdp) == 12
    assert (first_trajectories["available_capital"] >= -1e-9).all()
    pd.testing.assert_frame_equal(first_cmdp, second_cmdp)
    pd.testing.assert_frame_equal(first_trajectories, second_trajectories)


def test_portfolio_epsilon_greedy_explores_exploits_and_updates() -> None:
    encoder = BorrowerFeatureEncoder(include_protected_attributes=False)
    env = StatefulMicrofinanceCMDPEnv(
        CMDPEnvironmentConfig(
            initial_capital=100_000,
            max_rounds=3,
            pool_size=5,
            max_loans_per_round=1,
            repayment_delay_rounds=2,
            seed=4,
        ),
        encoder,
    )
    exploit = PortfolioAwareEpsilonGreedyPolicy(
        encoder.n_features, epsilon=0.0, seed=4
    )
    exploit_scores = exploit.scores(env)
    assert len(exploit_scores) == len(env.current_pool)
    context = env.current_pool[0].context.copy()
    before = exploit.model.theta.copy()
    exploit.update(context, reward=1.0, cost=env.current_pool[0].cost)
    assert exploit.total_updates == 1
    assert not np.array_equal(before, exploit.model.theta)

    explore = PortfolioAwareEpsilonGreedyPolicy(
        encoder.n_features, epsilon=1.0, seed=4
    )
    first = explore.scores(env)
    explore.reset()
    second = explore.scores(env)
    np.testing.assert_allclose(first, second)
    assert ((0.0 <= first) & (first <= 1.0)).all()


def aggregate_fixture(rows: list[dict[str, float | str]]) -> pd.DataFrame:
    metrics = (
        "cumulative_reward",
        "reward_per_cost",
        "repayment_rate",
        "demographic_parity_gap",
        "quota_satisfaction_rate",
        "total_quota_deficit",
        "budget_utilization",
    )
    records = []
    for row in rows:
        record = {("policy", ""): row["policy"]}
        for metric in metrics:
            record[(metric, "mean")] = row.get(metric, 0.0)
            record[(metric, "std")] = 0.0
        records.append(record)
    frame = pd.DataFrame(records)
    frame.columns = pd.MultiIndex.from_tuples(frame.columns)
    return frame.set_index(("policy", "")).rename_axis("policy")


def test_ranking_prefers_feasible_utility_and_has_deterministic_ties() -> None:
    aggregate = aggregate_fixture(
        [
            {
                "policy": "zeta",
                "cumulative_reward": 95.0,
                "reward_per_cost": 0.20,
                "repayment_rate": 0.8,
                "demographic_parity_gap": 0.04,
                "total_quota_deficit": 0.01,
            },
            {
                "policy": "alpha",
                "cumulative_reward": 95.0,
                "reward_per_cost": 0.20,
                "repayment_rate": 0.8,
                "demographic_parity_gap": 0.04,
                "total_quota_deficit": 0.01,
            },
            {
                "policy": "high_utility_infeasible",
                "cumulative_reward": 200.0,
                "reward_per_cost": 0.90,
                "repayment_rate": 0.9,
                "demographic_parity_gap": 0.30,
                "total_quota_deficit": 0.20,
            },
        ]
    )
    ranked = rank_results(aggregate, parity_tolerance=0.05)
    assert ranked.iloc[0]["policy"] == "alpha"
    assert bool(ranked.iloc[0]["fairness_feasible"])
    assert ranked.iloc[-1]["policy"] == "high_utility_infeasible"


def test_ranking_fallback_reports_best_compromise() -> None:
    aggregate = aggregate_fixture(
        [
            {
                "policy": "less_violation",
                "cumulative_reward": 10.0,
                "reward_per_cost": 0.10,
                "repayment_rate": 0.5,
                "demographic_parity_gap": 0.06,
                "total_quota_deficit": 0.02,
            },
            {
                "policy": "more_violation",
                "cumulative_reward": 100.0,
                "reward_per_cost": 1.0,
                "repayment_rate": 0.9,
                "demographic_parity_gap": 0.20,
                "total_quota_deficit": 0.20,
            },
        ]
    )
    ranked = rank_results(aggregate, parity_tolerance=0.05)
    assert ranked.iloc[0]["policy"] == "less_violation"
    assert "best compromise" in winner_summary(ranked, "contextual")


def test_reports_include_grid_manifest_html_and_combined_links(tmp_path) -> None:
    aggregate = aggregate_fixture(
        [
            {
                "policy": "winner",
                "cumulative_reward": 10.0,
                "reward_per_cost": 0.1,
                "repayment_rate": 0.8,
                "demographic_parity_gap": 0.02,
                "total_quota_deficit": 0.0,
                "budget_utilization": 0.9,
            }
        ]
    )
    contextual_output = tmp_path / "contextual"
    ranked = save_evaluation_report(
        aggregate,
        contextual_output,
        "contextual",
        parity_tolerance=0.05,
        seeds=[0, 1],
        configuration={"rounds": 3},
        policy_parameters={"epsilon": 0.1},
    )
    grid = format_ranked_table(ranked, "contextual")
    assert grid.startswith("+")
    assert "| Rank | Policy" in grid
    for filename in (
        "ranked_results.csv",
        "winner_summary.txt",
        "run_manifest.json",
        "report.html",
    ):
        assert (contextual_output / filename).exists()
    manifest = json.loads(
        (contextual_output / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["seeds"] == [0, 1]
    assert manifest["policy_parameters"]["epsilon"] == 0.1
    assert "winner" in (contextual_output / "report.html").read_text(
        encoding="utf-8"
    )

    combined = save_combined_report(
        {"contextual": ranked, "cmdp": ranked},
        tmp_path,
        arm_aggregate=aggregate,
        sweep_report=tmp_path / "cmdp_robustness" / "report.html",
    )
    combined_text = combined.read_text(encoding="utf-8")
    assert "no cross-mode winner" in combined_text
    assert "Mode 1 — Arm Bandits" in combined_text
    assert "Mode 4 — Phase 3 CMDP Robustness Sweeps" in combined_text
    assert "combined_aggregate_results.csv" in combined_text
    assert "run_manifest.json" in combined_text
    assert "contextual/report.html" in combined_text
    assert "cmdp/report.html" in combined_text
    assert "cmdp_robustness/report.html" in combined_text
