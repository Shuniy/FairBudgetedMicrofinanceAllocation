from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np


@dataclass(frozen=True, slots=True)
class CMDPEnvironmentConfig:
    """Configuration for the Phase-3 stateful lending environment.

    `initial_capital` is a stock of lender capital, not a one-shot cumulative
    spending budget. Funding a loan moves capital from available cash to
    capital-at-risk; a successful repayment later recycles principal back into
    available cash. Defaults permanently reduce the surviving capital stock.
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

    # Applicant population controls; kept consistent with Phase 2.
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
        for name in (
            "female_probability",
            "marginalized_probability",
            "reserved_probability",
            "disability_probability",
            "rural_probability",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class LagrangianConfig:
    """Dual-update parameters for soft CMDP fairness constraints."""

    target_shares: Mapping[str, float] = field(default_factory=dict)
    parity_tolerance: float = 0.05
    quota_tolerance: float = 0.01
    dual_learning_rate: float = 0.20
    initial_lambda_quota: float = 0.20
    initial_lambda_parity: float = 0.20
    max_lambda: float = 8.0
    warmup_rounds: int = 5

    def __post_init__(self) -> None:
        if any(not 0.0 <= v <= 1.0 for v in self.target_shares.values()):
            raise ValueError("target shares must lie in [0, 1]")
        if sum(self.target_shares.values()) > 1.0 + 1e-12:
            raise ValueError("target shares cannot sum to more than one")
        if not 0.0 <= self.parity_tolerance <= 1.0:
            raise ValueError("parity_tolerance must lie in [0, 1]")
        if self.quota_tolerance < 0:
            raise ValueError("quota_tolerance cannot be negative")
        if self.dual_learning_rate <= 0:
            raise ValueError("dual_learning_rate must be positive")
        if self.max_lambda <= 0:
            raise ValueError("max_lambda must be positive")
        if self.warmup_rounds < 0:
            raise ValueError("warmup_rounds cannot be negative")


@dataclass(frozen=True, slots=True)
class PendingLoan:
    applicant_id: str
    fairness_group: str
    principal: float
    funded_round: int
    due_round: int
    reward: float
    true_repayment_probability: float
    learning_context: np.ndarray

    @property
    def repaid(self) -> bool:
        return self.reward >= 0.5


@dataclass(frozen=True, slots=True)
class MaturedLoanFeedback:
    applicant_id: str
    fairness_group: str
    principal: float
    reward: float
    learning_context: np.ndarray
    capital_returned: float
    due_round: int


@dataclass(frozen=True, slots=True)
class FundingResult:
    round_index: int
    applicant_id: str
    fairness_group: str
    principal: float
    hidden_reward: float
    true_repayment_probability: float
    due_round: int
    available_capital: float
    capital_at_risk: float


@dataclass(frozen=True, slots=True)
class RoundAdvanceResult:
    new_round_index: int
    matured_feedback: tuple[MaturedLoanFeedback, ...]
    available_capital: float
    capital_at_risk: float
    done: bool
