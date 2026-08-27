from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from phase2.environment import ContextualMicrofinanceEnv
from phase2.feature_encoder import BorrowerFeatureEncoder
from phase2.metrics import (
    allocation_shares,
    budget_utilization,
    demographic_parity_gap,
    group_funding_rates,
    quota_deficits,
    quota_satisfaction_rate,
    repayment_rate,
    reward_per_unit_cost,
)
from phase2.models import ContextualEnvironmentConfig
from phase2.policies import ContextualPolicy


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    summary: dict[str, float | int | str]
    trajectory: pd.DataFrame


def run_episode(
    env: ContextualMicrofinanceEnv,
    policy: ContextualPolicy,
    target_shares: Mapping[str, float] | None = None,
    seed: int | None = None,
) -> EpisodeResult:
    targets = dict(target_shares or {})
    env.reset(seed)
    policy.seed = env.config.seed if seed is None else seed
    policy.reset()

    rows: list[dict[str, float | int | str]] = []
    cumulative_reward = 0.0
    cumulative_cost = 0.0

    while not env.done:
        action = policy.select_action(env)
        selected_candidate = env.current_pool[action]
        result = env.step(action)

        # The selected context is saved before the pool changes.
        for feedback in result.observed_feedback:
            policy.update(feedback.context, feedback.reward, feedback.cost)

        cumulative_reward += result.reward
        cumulative_cost += result.cost
        shares = allocation_shares(result.funding_counts)
        rates = group_funding_rates(
            result.funding_counts,
            result.opportunity_counts,
        )
        deficits = quota_deficits(result.funding_counts, targets)

        row: dict[str, float | int | str] = {
            "round": result.round_index,
            "applicant_id": result.applicant_id,
            "fairness_group": result.fairness_group,
            "reward": result.reward,
            "cost": result.cost,
            "true_repayment_probability": result.true_repayment_probability,
            "cumulative_reward": cumulative_reward,
            "cumulative_cost": cumulative_cost,
            "budget_remaining": result.budget_remaining,
            "budget_utilization": budget_utilization(
                env.config.total_budget, result.budget_remaining
            ),
            "reward_per_cost": cumulative_reward / cumulative_cost,
            "repayment_rate": cumulative_reward / (result.round_index + 1),
            "demographic_parity_gap": demographic_parity_gap(
                result.funding_counts,
                result.opportunity_counts,
            ),
            "quota_satisfaction_rate": quota_satisfaction_rate(
                result.funding_counts,
                targets,
            ),
            "observed_feedback_count": len(result.observed_feedback),
            "poverty_relief_score": selected_candidate.borrower.poverty_relief_score,
            "vulnerability_score": selected_candidate.borrower.vulnerability_score,
        }
        for group in sorted(set(result.opportunity_counts) | set(targets)):
            row[f"funded_{group}"] = result.funding_counts.get(group, 0)
            row[f"opportunities_{group}"] = result.opportunity_counts.get(group, 0)
            row[f"allocation_share_{group}"] = shares.get(group, 0.0)
            row[f"funding_rate_{group}"] = rates.get(group, 0.0)
            row[f"quota_deficit_{group}"] = deficits.get(group, 0.0)
        rows.append(row)

    trajectory = pd.DataFrame(rows)
    rewards = trajectory["reward"].to_numpy() if not trajectory.empty else np.array([])
    costs = trajectory["cost"].to_numpy() if not trajectory.empty else np.array([])
    final_shares = allocation_shares(env.group_funding_counts)
    final_rates = group_funding_rates(
        env.group_funding_counts,
        env.group_opportunity_counts,
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
            env.group_funding_counts,
            env.group_opportunity_counts,
        ),
        "quota_satisfaction_rate": quota_satisfaction_rate(
            env.group_funding_counts,
            targets,
        ),
        "pending_feedback": len(env.pending_feedback),
    }
    for group in sorted(env.group_opportunity_counts):
        summary[f"funded_{group}"] = env.group_funding_counts.get(group, 0)
        summary[f"opportunities_{group}"] = env.group_opportunity_counts.get(group, 0)
        summary[f"allocation_share_{group}"] = final_shares.get(group, 0.0)
        summary[f"funding_rate_{group}"] = final_rates.get(group, 0.0)
    return EpisodeResult(summary=summary, trajectory=trajectory)


def compare_policies(
    env_config: ContextualEnvironmentConfig,
    policy_factories: Mapping[str, Callable[[int, int], ContextualPolicy]],
    target_shares: Mapping[str, float],
    seeds: Sequence[int],
    encoder: BorrowerFeatureEncoder | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    encoder = encoder or BorrowerFeatureEncoder()
    summaries: list[dict[str, float | int | str]] = []
    trajectories: list[pd.DataFrame] = []

    for policy_name, factory in policy_factories.items():
        for seed in seeds:
            config = ContextualEnvironmentConfig(
                total_budget=env_config.total_budget,
                max_rounds=env_config.max_rounds,
                pool_size=env_config.pool_size,
                seed=seed,
                feedback_delay_rounds=env_config.feedback_delay_rounds,
                minimum_cost=env_config.minimum_cost,
                female_probability=env_config.female_probability,
                marginalized_probability=env_config.marginalized_probability,
                reserved_probability=env_config.reserved_probability,
                disability_probability=env_config.disability_probability,
                rural_probability=env_config.rural_probability,
            )
            env = ContextualMicrofinanceEnv(config=config, encoder=encoder)
            policy = factory(encoder.n_features, seed)
            episode = run_episode(env, policy, target_shares=target_shares, seed=seed)

            summary = dict(episode.summary)
            summary.update({"policy": policy_name, "seed": seed})
            summaries.append(summary)

            trajectory = episode.trajectory.copy()
            trajectory["policy"] = policy_name
            trajectory["seed"] = seed
            trajectories.append(trajectory)

    return pd.DataFrame(summaries), pd.concat(trajectories, ignore_index=True)
