from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from collections.abc import Mapping
import numpy as np


@dataclass(frozen=True, slots=True)
class FeedbackEvent:
    """Represents feedback received after taking an action in a round."""
    due_round: int
    reward: float
    cost: float
    action: int | None = None
    context: np.ndarray | None = None
    applicant_id: str | None = None


@dataclass(frozen=True, slots=True)
class StepResult:
    """Represents the result of taking an action in a round."""
    round_index: int
    selected_index: int
    identifier: str
    group: str
    reward: float
    cost: float
    budget_remaining: float
    observed_feedback: tuple[FeedbackEvent, ...]
    funding_counts: Mapping[str, int]
    opportunity_counts: Mapping[str, int]
    done: bool
    true_repayment_probability: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    population_opportunity_counts: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PendingLoan:
    """A Phase 3 loan whose outcome is hidden until its due round."""

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
    observed_feedback: tuple[MaturedLoanFeedback, ...] = ()


@dataclass(frozen=True, slots=True)
class RoundAdvanceResult:
    new_round_index: int
    matured_feedback: tuple[MaturedLoanFeedback, ...]
    available_capital: float
    capital_at_risk: float
    done: bool
