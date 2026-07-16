from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from Data.borrower import Borrower


@dataclass(frozen=True, slots=True)
class BorrowerCandidate:
    """Represents a candidate borrower with associated context and repayment probability."""
    borrower: Borrower
    context: np.ndarray
    true_repayment_probability: float

    @property
    def cost(self) -> float:
        return self.borrower.requested_loan_amount

    @property
    def fairness_group(self) -> str:
        return self.borrower.primary_fairness_group()
