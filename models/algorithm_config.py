from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AlgorithmConfig:
    """Configuration parameters for the algorithm. Need to check their validity once raw results are obtained. Tuning required"""
    epsilon: float = 0.10
    epsilon_decay: float = 0.02
    exploration_budget_fraction: float = 0.20
    exploration_coefficient: float = 2.0
    regularization: float = 1.0
    posterior_scale: float = 0.35
    budget_aware: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        if self.epsilon_decay <= 0:
            raise ValueError("epsilon_decay must be positive")
        if not 0.0 < self.exploration_budget_fraction < 1.0:
            raise ValueError("exploration_budget_fraction must be between 0 and 1")
        if self.exploration_coefficient < 0:
            raise ValueError("exploration_coefficient cannot be negative")
        if self.regularization <= 0 or self.posterior_scale <= 0:
            raise ValueError("regularization and posterior_scale must be positive")
