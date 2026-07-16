from __future__ import annotations

from dataclasses import dataclass

from Helpers.enums import Gender, LoanPurpose, Region, RegionType, Sector


@dataclass(frozen=True, slots=True)
class ApplicationContext:
    """information available when a loan application arrives. usually filled by applicant"""

    applicant_id: str
    name: str
    age: int
    gender: Gender # will be used to determine if the applicant is part of a protected group
    income: float # will be used to determine if the applicant is part of a protected group
    region_name: str
    region: str | Region
    loan_purpose: LoanPurpose # will be used to determine if the applicant is part of a protected group
    sector: Sector # will be used to determine if the applicant is part of a protected group
    region_type: RegionType
    repeated_borrower: bool # might use in cmdp or not sure about it yet
    urgent_need: bool # can be used in context, might be used in cmdp or not sure about it yet
    requested_loan_amount: float
    existing_debt: float
    previous_repayment_rate: float # can be used, but not sure to put it here, should be in borrower
    dependants: int = 0
    disability: bool = False # can be used to determine if the applicant is part of a protected group
    marginalized_community: bool = False # can be used to determine if the applicant is part of a protected group
    reserved_by_government: bool = False # can be used to determine if the applicant is part of a protected group
    medical_condition: bool = False # can be used to determine if the applicant is part of a protected group
    married: bool = False # dont know what to do with it

    def __post_init__(self) -> None:
        """validate the application context"""
        if not self.applicant_id.strip():
            raise ValueError("applicant_id cannot be empty")
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
