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
