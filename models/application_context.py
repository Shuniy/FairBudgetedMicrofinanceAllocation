from dataclasses import dataclass

from Helpers.enums import (
    Gender,
    LoanPurpose,
    Region,
    RegionType,
    SectorType,
)


@dataclass(frozen=True, slots=True)
class ApplicationContext:
    """
    These are borrower/application features
    features like:

    low income
    socially marginalized
    geographically isolated
    poverty alleviation
    social impact
    indebtedness risk
    """

    applicant_id: str
    age: int  # age of the borrower
    name: str  # name of the borrower
    disability: bool  # is the borrower disabled
    medical_condition: bool  # does the borrower have a medical condition

    # Core lending/application fields
    # requested_loan_amount: float  # loan amount requested by the borrower
    loan_purpose: LoanPurpose  # e.g. startup, working capital, business expansion
    sector: (
        SectorType  # e.g. agriculture, manufacturing, services, retail, personal, other
    )
    income: float  # annual income of the borrower // Taking GBP for reference, should be converted to local currency
    region_name: str  # e.g. city, town, village name
    region: Region  # north, south, east, west, central
    region_type: RegionType  # e.g. urban, rural, isolated
    repeated_borrower: bool  # has the borrower applied before
    gender: Gender
    marginalized_community: bool  # is the borrower from a minority group
    married: bool  # is the borrower married
    reserved_by_government: bool  # is the borrower from a government reserved category
    dependants: int # has dependants or not
    existing_debt: float # existing debt amount
    requested_loan_amount: float # amount requested

    # Socio-economic / protected attributes
    urgent_need: bool  # e.g. urgent need for loan
