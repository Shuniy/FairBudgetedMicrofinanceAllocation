from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from models.borrower import Borrower


@dataclass(frozen=True, slots=True)
class ContextualEnvironmentConfig:
    total_budget: float = 10_000.0
    max_rounds: int = 200
    pool_size: int = 20
    seed: int = 0
    feedback_delay_rounds: int = 0
    minimum_cost: float = 1.0

    # Applicant population controls.
    female_probability: float = 0.50
    marginalized_probability: float = 0.20
    reserved_probability: float = 0.12
    disability_probability: float = 0.06
    rural_probability: float = 0.35

    def __post_init__(self) -> None:
        if self.total_budget <= 0:
            raise ValueError("total_budget must be positive")
        if self.max_rounds <= 0:
            raise ValueError("max_rounds must be positive")
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
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class FairnessConfig:
    target_shares: Mapping[str, float] = field(default_factory=dict)
    demographic_parity_tolerance: float = 1.0
    penalty_lambda: float = 0.0

    def __post_init__(self) -> None:
        if any(not 0.0 <= value <= 1.0 for value in self.target_shares.values()):
            raise ValueError("target shares must be in [0, 1]")
        if sum(self.target_shares.values()) > 1.0 + 1e-12:
            raise ValueError("target shares cannot sum to more than one")
        if not 0.0 <= self.demographic_parity_tolerance <= 1.0:
            raise ValueError("demographic_parity_tolerance must be in [0, 1]")
        if self.penalty_lambda < 0:
            raise ValueError("penalty_lambda cannot be negative")


@dataclass(frozen=True, slots=True)
class BorrowerCandidate:
    borrower: Borrower
    context: np.ndarray
    true_repayment_probability: float

    @property
    def cost(self) -> float:
        return self.borrower.requested_loan_amount

    @property
    def fairness_group(self) -> str:
        return self.borrower.primary_fairness_group()


@dataclass(frozen=True, slots=True)
class FeedbackEvent:
    due_round: int
    context: np.ndarray
    reward: float
    cost: float
    applicant_id: str


@dataclass(frozen=True, slots=True)
class ContextualStepResult:
    round_index: int
    selected_index: int
    applicant_id: str
    fairness_group: str
    reward: float
    cost: float
    true_repayment_probability: float
    budget_remaining: float
    observed_feedback: tuple[FeedbackEvent, ...]
    funding_counts: Mapping[str, int]
    opportunity_counts: Mapping[str, int]
    done: bool
