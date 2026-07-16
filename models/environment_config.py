from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EnvironmentConfig:
    """Configuration parameters for the environment. will change as per required"""
    total_budget: float
    max_rounds: int
    seed: int = 0
    minimum_cost: float = 0.01

    def __post_init__(self) -> None:
        if self.total_budget <= 0:
            raise ValueError("total_budget must be strictly positive")
        if self.max_rounds <= 0:
            raise ValueError("max_rounds must be strictly positive")
        if self.minimum_cost <= 0:
            raise ValueError("minimum_cost must be strictly positive")


@dataclass(frozen=True, slots=True)
class ContextualEnvironmentConfig(EnvironmentConfig):
    pool_size: int = 20
    feedback_delay_rounds: int = 0
    female_probability: float = 0.50
    marginalized_probability: float = 0.20
    reserved_probability: float = 0.12
    disability_probability: float = 0.06
    rural_probability: float = 0.35

    def __post_init__(self) -> None:
        super(ContextualEnvironmentConfig, self).__post_init__()
        if self.pool_size <= 0:
            raise ValueError("pool_size must be positive")
        if self.feedback_delay_rounds < 0:
            raise ValueError("feedback_delay_rounds cannot be negative")
        for name in (
            "female_probability",
            "marginalized_probability",
            "reserved_probability",
            "disability_probability",
            "rural_probability",
        ):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.rural_probability > 0.85:
            raise ValueError("rural_probability cannot exceed 0.85")
