from __future__ import annotations

from dataclasses import dataclass


POPULATION_PROBABILITY_FIELDS = (
    "female_probability",
    "marginalized_probability",
    "reserved_probability",
    "disability_probability",
    "rural_probability",
)


def _validate_population_probabilities(config: object) -> None:
    """Validate the synthetic population controls shared by Phases 2 and 3."""

    for name in POPULATION_PROBABILITY_FIELDS:
        if not 0.0 <= getattr(config, name) <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    # The generator reserves probability 0.15 for semi-urban applicants.
    if getattr(config, "rural_probability") > 0.85:
        raise ValueError("rural_probability cannot exceed 0.85")


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
        _validate_population_probabilities(self)


@dataclass(frozen=True, slots=True)
class CMDPEnvironmentConfig:
    """Configuration for Phase 3's delayed, capital-recycling environment.

    Unlike ``EnvironmentConfig.total_budget``, ``initial_capital`` is a stock:
    funding moves cash to capital-at-risk and matured loans can return it.
    """

    initial_capital: float = 600_000.0
    max_rounds: int = 200
    pool_size: int = 20
    max_loans_per_round: int = 3
    repayment_delay_rounds: int = 8
    recycle_fraction: float = 1.0
    default_recovery_fraction: float = 0.0
    minimum_liquidity_reserve: float = 0.0
    minimum_cost: float = 1_000.0
    seed: int = 0
    female_probability: float = 0.50
    marginalized_probability: float = 0.20
    reserved_probability: float = 0.12
    disability_probability: float = 0.06
    rural_probability: float = 0.35

    def __post_init__(self) -> None:
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if self.max_rounds <= 0:
            raise ValueError("max_rounds must be positive")
        if self.pool_size <= 0:
            raise ValueError("pool_size must be positive")
        if self.max_loans_per_round <= 0:
            raise ValueError("max_loans_per_round must be positive")
        if self.repayment_delay_rounds < 1:
            raise ValueError("repayment_delay_rounds must be at least one in Phase 3")
        if not 0.0 <= self.recycle_fraction <= 1.0:
            raise ValueError("recycle_fraction must be in [0, 1]")
        if not 0.0 <= self.default_recovery_fraction <= 1.0:
            raise ValueError("default_recovery_fraction must be in [0, 1]")
        if self.minimum_liquidity_reserve < 0:
            raise ValueError("minimum_liquidity_reserve cannot be negative")
        if self.minimum_cost <= 0:
            raise ValueError("minimum_cost must be positive")
        _validate_population_probabilities(self)

    def borrower_generation_config(self, seed: int | None = None) -> ContextualEnvironmentConfig:
        """Adapt Phase 3 population settings to the shared borrower generator."""

        return ContextualEnvironmentConfig(
            total_budget=self.initial_capital,
            max_rounds=self.max_rounds,
            pool_size=self.pool_size,
            seed=self.seed if seed is None else seed,
            feedback_delay_rounds=0,
            minimum_cost=self.minimum_cost,
            female_probability=self.female_probability,
            marginalized_probability=self.marginalized_probability,
            reserved_probability=self.reserved_probability,
            disability_probability=self.disability_probability,
            rural_probability=self.rural_probability,
        )
