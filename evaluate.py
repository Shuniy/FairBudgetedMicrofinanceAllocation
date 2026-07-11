from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from Algorithms.bandit import BanditPolicy
from Environments.synthetic_environment_bandits import SyntheticEnvironment
from models.arm_spec import ArmSpec
from models.environment_config import EnvironmentConfig

from metrics import (
    allocation_shares,
    budget_utilization,
    demographic_parity_gap,
    quota_deficits,
    quota_satisfaction_rate,
    reward_per_unit_cost,
)


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    summary: dict[str, float | int | str]
    trajectory: pd.DataFrame


def run_episode(
    env: SyntheticEnvironment,
    policy: BanditPolicy,
    target_shares: Mapping[str, float] | None = None,
    seed: int | None = None,
) -> EpisodeResult:
    target_shares = dict(target_shares or {})
    env.reset(seed=seed)
    policy.reset()

    rows: list[dict[str, float | int | str]] = []
    cumulative_reward = 0.0
    cumulative_cost = 0.0

    while not env.done:
        action = policy.select_action(env)
        result = env.step(action)
        policy.update(action, result.reward, result.cost)
        cumulative_reward += result.reward
        cumulative_cost += result.cost

        shares = allocation_shares(env.group_allocation_counts)
        deficits = quota_deficits(env.group_allocation_counts, target_shares)
        row: dict[str, float | int | str] = {
            "round": result.round_index,
            "action": result.action,
            "arm": result.arm_name,
            "group": result.group,
            "reward": result.reward,
            "cost": result.cost,
            "cumulative_reward": cumulative_reward,
            "cumulative_cost": cumulative_cost,
            "budget_remaining": result.budget_remaining,
            "budget_utilization": budget_utilization(
                env.config.total_budget, result.budget_remaining
            ),
            "reward_per_cost": cumulative_reward / cumulative_cost,
            "demographic_parity_gap": demographic_parity_gap(
                env.group_allocation_counts
            ),
            "quota_satisfaction_rate": quota_satisfaction_rate(
                env.group_allocation_counts, target_shares
            ),
        }
        for group in env.groups:
            row[f"count_{group}"] = env.group_allocation_counts[group]
            row[f"share_{group}"] = shares[group]
            row[f"quota_deficit_{group}"] = deficits.get(group, 0.0)
        rows.append(row)

    trajectory = pd.DataFrame(rows)
    rewards = trajectory["reward"].to_numpy() if not trajectory.empty else np.array([])
    costs = trajectory["cost"].to_numpy() if not trajectory.empty else np.array([])
    final_shares = allocation_shares(env.group_allocation_counts)
    summary: dict[str, float | int | str] = {
        "rounds": env.total_allocations,
        "cumulative_reward": float(np.sum(rewards)),
        "cumulative_cost": float(np.sum(costs)),
        "reward_per_cost": reward_per_unit_cost(rewards, costs),
        "budget_utilization": budget_utilization(
            env.config.total_budget, env.remaining_budget
        ),
        "demographic_parity_gap": demographic_parity_gap(env.group_allocation_counts),
        "quota_satisfaction_rate": quota_satisfaction_rate(
            env.group_allocation_counts, target_shares
        ),
    }
    for group in env.groups:
        summary[f"allocations_{group}"] = env.group_allocation_counts[group]
        summary[f"share_{group}"] = final_shares[group]
    return EpisodeResult(summary=summary, trajectory=trajectory)


def compare_policies(
    arms: Sequence[ArmSpec],
    env_config: EnvironmentConfig,
    policy_factories: Mapping[str, Callable[[int, int], BanditPolicy]],
    target_shares: Mapping[str, float],
    seeds: Sequence[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries: list[dict[str, float | int | str]] = []
    trajectories: list[pd.DataFrame] = []

    for policy_name, factory in policy_factories.items():
        for seed in seeds:
            env = SyntheticEnvironment(
                arms=arms,
                config=EnvironmentConfig(
                    total_budget=env_config.total_budget,
                    max_rounds=env_config.max_rounds,
                    seed=seed,
                    minimum_cost=env_config.minimum_cost,
                ),
            )
            policy = factory(len(arms), seed)
            episode = run_episode(env, policy, target_shares=target_shares, seed=seed)
            summary = dict(episode.summary)
            summary.update({"policy": policy_name, "seed": seed})
            summaries.append(summary)

            trajectory = episode.trajectory.copy()
            trajectory["policy"] = policy_name
            trajectory["seed"] = seed
            trajectories.append(trajectory)

    return pd.DataFrame(summaries), pd.concat(trajectories, ignore_index=True)
