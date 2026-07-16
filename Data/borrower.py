from __future__ import annotations

from dataclasses import dataclass, field

from Helpers.enums import Gender, IncomeGroup, ProtectedGroup, Region, RegionType
from models.application_context import ApplicationContext


@dataclass(slots=True)
class Borrower:
    """Borrower domain object with validated, derived need features."""

    application_context: ApplicationContext
    protected_groups: set[ProtectedGroup] = field(init=False)
    income_group: IncomeGroup = field(init=False)
    socially_marginalized: bool = field(init=False)
    geographically_isolated: bool = field(init=False)
    poverty_relief_score: float = field(init=False)
    vulnerability_score: float = field(init=False)
    id: str = field(init=False)
    name: str = field(init=False)
    age: int = field(init=False)
    gender: Gender = field(init=False)
    income: float = field(init=False)
    region_name: str = field(init=False)
    region: str | Region = field(init=False)
    loan_purpose: object = field(init=False)
    sector: object = field(init=False)
    region_type: RegionType = field(init=False)
    repeated_borrower: bool = field(init=False)
    urgent_need: bool = field(init=False)

    def __post_init__(self) -> None:
        application = self.application_context
        self.id = application.applicant_id
        self.name = application.name
        self.age = application.age
        self.gender = application.gender
        self.income = application.income
        self.region_name = application.region_name
        self.region = application.region
        self.loan_purpose = application.loan_purpose
        self.sector = application.sector
        self.region_type = application.region_type
        self.repeated_borrower = application.repeated_borrower
        self.urgent_need = application.urgent_need
        self.calculate_scores_features()

    @property
    def requested_loan_amount(self) -> float:
        return self.application_context.requested_loan_amount

    @property
    def existing_debt(self) -> float:
        return self.application_context.existing_debt

    @property
    def previous_repayment_rate(self) -> float:
        return self.application_context.previous_repayment_rate

    def is_geographically_isolated(self) -> bool:
        return self.region_type in {RegionType.ISOLATED, RegionType.RURAL}

    def find_protected_groups(self) -> set[ProtectedGroup]:
        groups: set[ProtectedGroup] = set()
        application = self.application_context
        if self.gender == Gender.FEMALE:
            groups.add(ProtectedGroup.WOMEN)
        if application.disability:
            groups.add(ProtectedGroup.DISABLED)
        if application.age >= 60:
            groups.add(ProtectedGroup.AGED)
        if application.marginalized_community:
            groups.add(ProtectedGroup.MINORITY)
        if application.reserved_by_government:
            groups.add(ProtectedGroup.RESERVED)
        return groups or {ProtectedGroup.NONE}

    def is_socially_marginalized(self) -> bool:
        return bool(
            self.protected_groups
            & {ProtectedGroup.RESERVED, ProtectedGroup.MINORITY}
        )

    def find_income_group(self) -> IncomeGroup:
        if self.income < 10_000:
            return IncomeGroup.VERY_LOW
        if self.income < 20_000:
            return IncomeGroup.LOW
        if self.income < 40_000:
            return IncomeGroup.LOWER_MIDDLE
        if self.income < 80_000:
            return IncomeGroup.MIDDLE
        if self.income < 150_000:
            return IncomeGroup.UPPER_MIDDLE
        return IncomeGroup.HIGH

    def calculate_poverty_relief_score(self) -> float:
        income_need = 1.0 - min(self.income / 150_000.0, 1.0)
        geographic_need = 0.15 if self.geographically_isolated else 0.0
        marginalization_need = 0.20 if self.socially_marginalized else 0.0
        disability_need = 0.10 if self.application_context.disability else 0.0
        urgency_need = 0.10 if self.urgent_need else 0.0
        dependant_need = min(self.application_context.dependants / 8.0, 1.0) * 0.10
        return float(
            min(
                1.0,
                0.55 * income_need
                + geographic_need
                + marginalization_need
                + disability_need
                + urgency_need
                + dependant_need,
            )
        )

    def calculate_vulnerability_score(self) -> float:
        protected_count = len(self.protected_groups - {ProtectedGroup.NONE})
        return float(
            min(
                1.0,
                0.60 * self.poverty_relief_score
                + 0.10 * min(protected_count, 3)
                + (0.10 if self.geographically_isolated else 0.0),
            )
        )

    def primary_fairness_group(self) -> str:
        for group in (
            ProtectedGroup.DISABLED,
            ProtectedGroup.RESERVED,
            ProtectedGroup.MINORITY,
            ProtectedGroup.WOMEN,
            ProtectedGroup.AGED,
        ):
            if group in self.protected_groups:
                return group.value
        return ProtectedGroup.NONE.value

    def calculate_scores_features(self) -> None:
        self.protected_groups = self.find_protected_groups()
        self.income_group = self.find_income_group()
        self.socially_marginalized = self.is_socially_marginalized()
        self.geographically_isolated = self.is_geographically_isolated()
        self.poverty_relief_score = self.calculate_poverty_relief_score()
        self.vulnerability_score = self.calculate_vulnerability_score()
