from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np

from .models import ArmSpec, EnvironmentConfig


@dataclass(frozen=True, slots=True)
class StepResult:
    round_index: int
    action: int
    arm_name: str
    group: str
    reward: float
    cost: float
    budget_remaining: float
    done: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class SyntheticMicrofinanceEnv:
    """Small group-level microfinance environment for Phase 1.

    At the start of each round, one loan request cost is visible for every arm.
    The policy chooses a feasible arm, the environment samples a binary
    repayment reward, deducts the known cost, and generates a new set of costs
    for the next round.

    Revealing cost before selection matches lending practice: the requested
    loan amount is known when the allocation decision is made.
    """

    def __init__(self, arms: Sequence[ArmSpec], config: EnvironmentConfig) -> None:
        if not arms:
            raise ValueError("At least one arm is required")
        self.arms = tuple(arms)
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self._groups = tuple(dict.fromkeys(arm.group for arm in self.arms))
        self.reset()

    @property
    def n_arms(self) -> int:
        return len(self.arms)

    @property
    def groups(self) -> tuple[str, ...]:
        return self._groups

    @property
    def spent_budget(self) -> float:
        return self.config.total_budget - self.remaining_budget

    @property
    def total_allocations(self) -> int:
        return len(self.history)

    @property
    def done(self) -> bool:
        return self._done

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng(self.config.seed)

        self.remaining_budget = float(self.config.total_budget)
        self.round_index = 0
        self.history: list[dict[str, Any]] = []
        self.arm_allocation_counts = np.zeros(len(self.arms), dtype=int)
        self.group_allocation_counts = {group: 0 for group in self.groups}
        self.current_costs = self._sample_round_costs()
        self._done = False
        self._refresh_done_flag()
        return self.observation()

    def _sample_round_costs(self) -> np.ndarray:
        costs: list[float] = []
        for arm in self.arms:
            if arm.cost_std == 0:
                sampled = arm.cost_mean
            else:
                sampled = self.rng.normal(arm.cost_mean, arm.cost_std)
            costs.append(max(self.config.minimum_cost, float(sampled)))
        return np.asarray(costs, dtype=float)

    def feasible_actions(self) -> list[int]:
        if self.round_index >= self.config.max_rounds:
            return []
        return [
            index
            for index, cost in enumerate(self.current_costs)
            if cost <= self.remaining_budget + 1e-12
        ]

    def observation(self) -> dict[str, Any]:
        return {
            "round_index": self.round_index,
            "remaining_budget": self.remaining_budget,
            "spent_budget": self.spent_budget,
            "current_costs": self.current_costs.copy(),
            "feasible_actions": self.feasible_actions(),
            "arm_allocation_counts": self.arm_allocation_counts.copy(),
            "group_allocation_counts": dict(self.group_allocation_counts),
        }

    def step(self, action: int) -> StepResult:
        if self.done:
            raise RuntimeError("Cannot step a completed environment")
        if action < 0 or action >= self.n_arms:
            raise IndexError(f"Invalid action {action}")
        if action not in self.feasible_actions():
            raise ValueError(
                f"Action {action} is infeasible: cost={self.current_costs[action]:.3f}, "
                f"remaining_budget={self.remaining_budget:.3f}"
            )

        arm = self.arms[action]
        cost = float(self.current_costs[action])
        reward = float(self.rng.binomial(1, arm.repayment_probability))

        self.remaining_budget -= cost
        self.remaining_budget = max(0.0, self.remaining_budget)
        self.arm_allocation_counts[action] += 1
        self.group_allocation_counts[arm.group] += 1

        result = StepResult(
            round_index=self.round_index,
            action=action,
            arm_name=arm.name,
            group=arm.group,
            reward=reward,
            cost=cost,
            budget_remaining=self.remaining_budget,
            done=False,
        )
        self.history.append(result.as_dict())
        self.round_index += 1
        self.current_costs = self._sample_round_costs()
        self._refresh_done_flag()

        final_result = StepResult(
            round_index=result.round_index,
            action=result.action,
            arm_name=result.arm_name,
            group=result.group,
            reward=result.reward,
            cost=result.cost,
            budget_remaining=result.budget_remaining,
            done=self.done,
        )
        self.history[-1] = final_result.as_dict()
        return final_result

    def _refresh_done_flag(self) -> None:
        self._done = (
            self.round_index >= self.config.max_rounds
            or len(self.feasible_actions()) == 0
        )
