from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from phase2.feature_encoder import BorrowerFeatureEncoder
from phase3.environment import StatefulMicrofinanceCMDPEnv
from phase3.metrics import (
    allocation_shares,
    capital_survival_ratio,
    capital_turnover,
    demographic_parity_gap,
    group_funding_rates,
    liquidity_utilization,
    quota_deficits,
    quota_satisfaction_rate,
    repayment_rate,
    reward_per_disbursed_capital,
    total_quota_deficit,
)
from phase3.models import CMDPEnvironmentConfig
from phase3.policies import CMDPPolicy, LagrangianFairCMDPPolicy


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    summary: dict[str, float | int | str]
    trajectory: pd.DataFrame
    funding_log: pd.DataFrame
    maturity_log: pd.DataFrame


def _policy_lambdas(policy: CMDPPolicy) -> tuple[float, float]:
    if isinstance(policy, LagrangianFairCMDPPolicy):
        return policy.lambda_quota, policy.lambda_parity
    return 0.0, 0.0


def run_episode(
    env: StatefulMicrofinanceCMDPEnv,
    policy: CMDPPolicy,
    target_shares: Mapping[str, float],
    parity_tolerance: float,
    seed: int,
) -> EpisodeResult:
    env.reset(seed)
    policy.seed = seed
    policy.reset()

    rows: list[dict[str, float | int | str]] = []
    while not env.done:
        # Repayments due at this round were released by the previous advance.
        matured_now = env._matured_at_round_start
        for feedback in matured_now:
            policy.update(feedback.learning_context, feedback.reward, feedback.principal)

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
        lamb_q, lamb_dp = _policy_lambdas(policy)
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
                    env.cumulative_disbursed, env.config.initial_capital
                ),
                "liquidity_utilization": liquidity_utilization(
                    env.available_capital, env.capital_at_risk
                ),
                "capital_survival_ratio": capital_survival_ratio(
                    env.available_capital,
                    env.capital_at_risk,
                    env.config.initial_capital,
                ),
                "demographic_parity_gap": fairness["demographic_parity_gap"],
                "total_quota_deficit": fairness["total_quota_deficit"],
                "quota_satisfaction_rate": quota_satisfaction_rate(
                    env.group_funding_counts, target_shares
                ),
                "lambda_quota": lamb_q,
                "lambda_parity": lamb_dp,
                "state_bucket": env.state_bucket(target_shares, parity_tolerance),
            }
        )

        # Update dual variables from the fairness state produced by this round.
        policy.on_round_end(env)
        advance = env.advance_round()
        if advance.done:
            break

    # Outcomes are sampled at funding time. Eventual reward includes pending
    # loans for evaluation, while observed_reward_by_horizon only counts loans
    # that matured before the decision horizon ended.
    eventual_rewards = np.asarray(
        [float(row["hidden_reward"]) for row in env.funding_history], dtype=float
    )
    principals = np.asarray(
        [float(row["principal"]) for row in env.funding_history], dtype=float
    )
    eventual_terminal_capital = env.available_capital + env.eventual_pending_capital_return()
    final_rates = group_funding_rates(
        env.group_funding_counts, env.group_opportunity_counts
    )
    final_shares = allocation_shares(env.group_funding_counts)
    final_deficits = quota_deficits(env.group_funding_counts, target_shares)
    lambda_q, lambda_dp = _policy_lambdas(policy)

    summary: dict[str, float | int | str] = {
        "rounds": env.round_index,
        "loans_funded": env.total_allocations,
        "cumulative_reward": float(np.sum(eventual_rewards)),
        "observed_reward_by_horizon": float(env.observed_repayments),
        "repayment_rate": repayment_rate(eventual_rewards),
        "cumulative_disbursed": env.cumulative_disbursed,
        "reward_per_cost": reward_per_disbursed_capital(eventual_rewards, principals),
        "capital_turnover": capital_turnover(
            env.cumulative_disbursed, env.config.initial_capital
        ),
        "cumulative_recycled": env.cumulative_recycled,
        "default_loss_by_horizon": env.cumulative_default_loss,
        "final_available_capital": env.available_capital,
        "final_capital_at_risk": env.capital_at_risk,
        "eventual_terminal_capital": eventual_terminal_capital,
        "capital_survival_ratio": capital_survival_ratio(
            env.available_capital, env.capital_at_risk, env.config.initial_capital
        ),
        "demographic_parity_gap": demographic_parity_gap(
            env.group_funding_counts, env.group_opportunity_counts
        ),
        "quota_satisfaction_rate": quota_satisfaction_rate(
            env.group_funding_counts, target_shares
        ),
        "total_quota_deficit": total_quota_deficit(
            env.group_funding_counts, target_shares
        ),
        "pending_loans": len(env.pending_loans),
        "lambda_quota_final": lambda_q,
        "lambda_parity_final": lambda_dp,
    }
    for group in sorted(env.group_opportunity_counts):
        summary[f"funded_{group}"] = env.group_funding_counts.get(group, 0)
        summary[f"opportunities_{group}"] = env.group_opportunity_counts.get(group, 0)
        summary[f"allocation_share_{group}"] = final_shares.get(group, 0.0)
        summary[f"funding_rate_{group}"] = final_rates.get(group, 0.0)
        summary[f"quota_deficit_{group}"] = final_deficits.get(group, 0.0)

    return EpisodeResult(
        summary=summary,
        trajectory=pd.DataFrame(rows),
        funding_log=pd.DataFrame(env.funding_history),
        maturity_log=pd.DataFrame(env.maturity_history),
    )


def compare_policies(
    env_config: CMDPEnvironmentConfig,
    policy_factories: Mapping[str, Callable[[int, int], CMDPPolicy]],
    target_shares: Mapping[str, float],
    parity_tolerance: float,
    seeds: Sequence[int],
    encoder: BorrowerFeatureEncoder | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    encoder = encoder or BorrowerFeatureEncoder(include_protected_attributes=False)
    summaries: list[dict[str, float | int | str]] = []
    trajectories: list[pd.DataFrame] = []
    funding_logs: list[pd.DataFrame] = []
    maturity_logs: list[pd.DataFrame] = []

    for policy_name, factory in policy_factories.items():
        for seed in seeds:
            config = CMDPEnvironmentConfig(
                initial_capital=env_config.initial_capital,
                max_rounds=env_config.max_rounds,
                pool_size=env_config.pool_size,
                max_loans_per_round=env_config.max_loans_per_round,
                repayment_delay_rounds=env_config.repayment_delay_rounds,
                recycle_fraction=env_config.recycle_fraction,
                default_recovery_fraction=env_config.default_recovery_fraction,
                minimum_liquidity_reserve=env_config.minimum_liquidity_reserve,
                minimum_cost=env_config.minimum_cost,
                seed=seed,
                female_probability=env_config.female_probability,
                marginalized_probability=env_config.marginalized_probability,
                reserved_probability=env_config.reserved_probability,
                disability_probability=env_config.disability_probability,
                rural_probability=env_config.rural_probability,
            )
            env = StatefulMicrofinanceCMDPEnv(config=config, encoder=encoder)
            policy = factory(encoder.n_features, seed)
            episode = run_episode(
                env=env,
                policy=policy,
                target_shares=target_shares,
                parity_tolerance=parity_tolerance,
                seed=seed,
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
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    return (
        pd.DataFrame(summaries),
        concat_or_empty(trajectories),
        concat_or_empty(funding_logs),
        concat_or_empty(maturity_logs),
    )
