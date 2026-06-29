from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

from models.arm_spec import ArmSpec
from models.environment_config import EnvironmentConfig


@dataclass(frozen=True, slots=True)
class StepResult:
    """Represents the result of a single step in the environment."""

    round_index: int
    action: int
    arm_name: str
    group: str
    reward: float
    cost: float
    budget_remaining: float
    done: bool

    def as_dict(self):
        return asdict(self)


class SyntheticEnvironment:
    def __init__(self, arms: Sequence[ArmSpec], config: EnvironmentConfig):
        # need atleast one arm
        if not arms:
            raise ValueError("At least one arm is required")
        self.arms = arms
        self.config = config
        # need to use a random number generator for reproducibility
        self.rng = np.random.default_rng(seed=config.seed)
        # groups are unique, so we can use a dict to remove duplicates
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

    def reset(self, seed: int | None = None):
        """Reset the environment"""
        # check for seed
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng(self.config.seed)

        self.remaining_budget = float(self.config.total_budget)
        self.round_index = 0
        self.history = []
        self.arm_allocation_counts = np.zeros(len(self.arms), dtype=int)
        self.group_allocation_counts = {group: 0 for group in self.groups}
        self.current_costs = self.round_costs()
        self._done = False
        self._refresh_done_flag()
        return self.observation()

    def round_costs(self):
        # each round cost, just get a sample from the distribution made from mean and standard deviation
        costs: list[float] = []
        for arm in self.arms:
            if arm.cost_std == 0:
                sampled = arm.cost_mean
            else:
                sampled = self.rng.normal(arm.cost_mean, arm.cost_std)
            costs.append(max(self.config.minimum_cost, float(sampled)))
        return np.asarray(costs, dtype=float)

    def _refresh_done_flag(self):
        # set done flag if we have reached max rounds or if there are no feasible actions left
        self._done = (
            self.round_index >= self.config.max_rounds
            or len(self.feasible_actions()) == 0
        )

    def observation(self):
        # return current state of the environment
        return {
            "round_index": self.round_index,
            "remaining_budget": self.remaining_budget,
            "spent_budget": self.spent_budget,
            "current_costs": self.current_costs.copy(),
            "feasible_actions": self.feasible_actions(),
            "arm_allocation_counts": self.arm_allocation_counts.copy(),
            "group_allocation_counts": dict(self.group_allocation_counts),
        }

    def feasible_actions(self) -> list[int]:
        # feasible actions means groups that have costs less than or equal to the remaining budget
        # and we have not reached max rounds
        if self.round_index >= self.config.max_rounds:
            return []
        return [
            index
            for index, cost in enumerate(self.current_costs)
            if cost <= self.remaining_budget + 1e-12
        ]

    def step(self, action: int):
        # each step, we check if the action is valid and feasible,
        if self.done:
            raise RuntimeError("Cannot step a completed environment")
        if action < 0 or action >= self.n_arms:
            raise IndexError(f"Invalid action {action}")
        if action not in self.feasible_actions():
            raise ValueError(
                f"Action {action} is infeasible: cost={self.current_costs[action]:.3f}, "
                f"remaining_budget={self.remaining_budget:.3f}"
            )

        # then we sample the reward and cost,
        arm = self.arms[action]
        cost = float(self.current_costs[action])
        reward = float(self.rng.binomial(1, arm.repayment_probability))

        # update the environment state
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
        self.current_costs = self.round_costs()
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
