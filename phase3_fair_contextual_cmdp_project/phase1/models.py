from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True, slots=True)
class ArmSpec:
    """Parameters for one Phase-1 arm.

    In Phase 1, an arm represents a borrower group or borrower-type bucket.
    The repayment outcome is Bernoulli and the requested loan cost is sampled
    from a truncated normal distribution. A zero standard deviation gives a
    deterministic loan cost, which is useful for KUBE experiments.
    """

    name: str
    group: str
    repayment_probability: float
    cost_mean: float
    cost_std: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.repayment_probability <= 1.0:
            raise ValueError("repayment_probability must be in [0, 1]")
        if self.cost_mean <= 0:
            raise ValueError("cost_mean must be strictly positive")
        if self.cost_std < 0:
            raise ValueError("cost_std cannot be negative")


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


@dataclass(frozen=True, slots=True)
class FairnessConfig:
    """Running group-fairness settings.

    target_shares are lower-bound allocation quotas. They do not need to sum
    to one, but their sum cannot exceed one.
    """

    target_shares: Mapping[str, float] = field(default_factory=dict)
    demographic_parity_tolerance: float = 1.0
    penalty_lambda: float = 0.0

    def __post_init__(self) -> None:
        if any(not 0.0 <= value <= 1.0 for value in self.target_shares.values()):
            raise ValueError("Every target share must be in [0, 1]")
        if sum(self.target_shares.values()) > 1.0 + 1e-12:
            raise ValueError("The target shares cannot sum to more than 1")
        if not 0.0 <= self.demographic_parity_tolerance <= 1.0:
            raise ValueError("demographic_parity_tolerance must be in [0, 1]")
        if self.penalty_lambda < 0:
            raise ValueError("penalty_lambda cannot be negative")
