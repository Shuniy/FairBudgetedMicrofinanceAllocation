from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EnvironmentConfig:
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
