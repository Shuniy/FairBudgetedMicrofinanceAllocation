from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ArmSpec:
    """Distribution parameters for one borrower-group arm. Only for phase 1 as of now"""

    name: str
    group: str
    repayment_probability: float
    cost_mean: float
    cost_std: float = 0.0

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.group.strip():
            raise ValueError("arm name and group cannot be empty")
        if not 0.0 <= self.repayment_probability <= 1.0:
            raise ValueError("repayment_probability must be in [0, 1]")
        if self.cost_mean <= 0:
            raise ValueError("cost_mean must be strictly positive")
        if self.cost_std < 0:
            raise ValueError("cost_std cannot be negative")
