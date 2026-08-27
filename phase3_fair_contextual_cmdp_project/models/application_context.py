from __future__ import annotations

from dataclasses import dataclass

from Helpers.enums import Gender, LoanPurpose, RegionType, Sector


@dataclass(frozen=True, slots=True)
class ApplicationContext:
    """Observable borrower information available when an application arrives.

    Protected attributes are retained for fairness auditing. The default
    feature encoder deliberately excludes them from repayment prediction.
    """

    applicant_id: str
    name: str
    age: int
    gender: Gender
    income: float
    region_name: str
    region: str
    loan_purpose: LoanPurpose
    sector: Sector
    region_type: RegionType
    repeated_borrower: bool
    urgent_need: bool

    # Financial/contextual fields needed by the Phase-2 simulator.
    requested_loan_amount: float
    existing_debt: float
    previous_repayment_rate: float
    dependants: int = 0

    # Fairness-monitoring attributes.
    disability: bool = False
    marginalized_community: bool = False
    reserved_by_government: bool = False

    def __post_init__(self) -> None:
        if not 18 <= self.age <= 100:
            raise ValueError("age must be between 18 and 100")
        if self.income <= 0:
            raise ValueError("income must be strictly positive")
        if self.requested_loan_amount <= 0:
            raise ValueError("requested_loan_amount must be strictly positive")
        if self.existing_debt < 0:
            raise ValueError("existing_debt cannot be negative")
        if not 0.0 <= self.previous_repayment_rate <= 1.0:
            raise ValueError("previous_repayment_rate must be in [0, 1]")
        if self.dependants < 0:
            raise ValueError("dependants cannot be negative")
