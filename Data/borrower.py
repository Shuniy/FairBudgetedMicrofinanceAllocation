from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from Helpers.enums import Gender, IncomeGroup, ProtectedGroup, RegionType
from models.application_context import ApplicationContext


@dataclass
class Borrower:
    def __init__(self, application: ApplicationContext) -> None:
        self.application_context: ApplicationContext = application

        self.id: str = self.application_context.applicant_id
        self.name: str = self.application_context.name
        self.age: int = self.application_context.age
        self.gender = self.application_context.gender
        self.income: float = self.application_context.income
        self.region_name: str = self.application_context.region_name
        self.region = self.application_context.region
        self.loan_purpose = self.application_context.loan_purpose
        self.sector = self.application_context.sector
        self.region_type = self.application_context.region_type
        self.repeated_borrower: bool = self.application_context.repeated_borrower
        self.urgent_need: bool = self.application_context.urgent_need

    def is_geographically_isolated(self) -> bool:
        return self.region_type in [RegionType.ISOLATED, RegionType.RURAL]

    def find_protected_groups(self):
        groups = set()
        if self.gender == Gender.FEMALE:
            groups.add(ProtectedGroup.WOMEN)
        if self.application_context.disability:
            groups.add(ProtectedGroup.DISABLED)
        if self.application_context.age >= 60:
            groups.add(ProtectedGroup.AGED)
        if self.application_context.marginalized_community:
            groups.add(ProtectedGroup.MINORITY)
        if self.application_context.reserved_by_government:
            groups.add(ProtectedGroup.RESERVED)
        return groups

    def is_socially_marginalized(self) -> bool:
        return (
            len(
                self.protected_groups.intersection(
                    {ProtectedGroup.RESERVED, ProtectedGroup.MINORITY}
                )
            )
            > 0
        )

    def find_income_group(self) -> IncomeGroup:
        income = self.application_context.income
        if income < 10000:
            return IncomeGroup.VERY_LOW
        elif income < 20000:
            return IncomeGroup.LOW
        elif income < 40000:
            return IncomeGroup.LOWER_MIDDLE
        elif income < 80000:
            return IncomeGroup.MIDDLE
        elif income < 150000:
            return IncomeGroup.MIDDLE
        else:
            return IncomeGroup.HIGH

    def calculate_scores_features(self):
        self.protected_groups = self.find_protected_groups()
        self.income_group = self.find_income_group()
        self.socially_marginalized: Optional[bool] = self.is_socially_marginalized()
        self.geographically_isolated: Optional[bool] = self.is_geographically_isolated()

        # self.poverty_relief_score: Optional[float] = self.is_poverty_relief_eligible()
        # self.social_impact_score: Optional[float] = self.calculate_social_impact_score()
        # self.indebtedness_risk_score: Optional[float] = (
        #     self.calculate_indebtedness_risk_score()
        # )
        # self.priority_weight: Optional[float] = self.calculate_priority_weight()
        # self.historical_repayment_proxy: Optional[float] = None
        # self.vulnerability_score: Optional[float] = None
