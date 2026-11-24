from __future__ import annotations

import numpy as np

from Data.borrower import Borrower
from Helpers.enums import LoanPurpose, RegionType, Sector


class BorrowerFeatureEncoder:
    """Deterministic borrower vectorizer shared by contextual policies."""

    def __init__(self, include_protected_attributes: bool = False) -> None:
        self.include_protected_attributes = include_protected_attributes
        self.sectors = tuple(Sector)
        self.purposes = tuple(LoanPurpose)
        self.region_types = tuple(RegionType)
        names = [
            "intercept",
            "log_income_scaled",
            "loan_amount_scaled",
            "loan_to_income",
            "existing_debt_scaled",
            "debt_to_income",
            "previous_repayment_rate",
            "repeated_borrower",
            "urgent_need",
            "dependants_scaled",
        ]
        if include_protected_attributes:
            names.insert(1, "age_scaled")
            names += ["poverty_relief_score", "vulnerability_score"]
        names += [f"sector={value.value}" for value in self.sectors]
        names += [f"purpose={value.value}" for value in self.purposes]
        names += [f"region_type={value.value}" for value in self.region_types]
        if include_protected_attributes:
            names += [
                "gender_female",
                "disability",
                "marginalized",
                "reserved",
                "aged",
            ]
        self.feature_names = tuple(names)

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    def transform(self, borrower: Borrower) -> np.ndarray:
        application = borrower.application_context
        income = max(application.income, 1.0)
        loan = application.requested_loan_amount
        debt = application.existing_debt
        values: list[float] = [
            1.0,
            float(np.clip(np.log1p(income) / np.log1p(200_000.0), 0.0, 1.0)),
            float(np.clip(loan / 50_000.0, 0.0, 1.0)),
            float(np.clip(loan / income, 0.0, 2.0) / 2.0),
            float(np.clip(debt / 100_000.0, 0.0, 1.0)),
            float(np.clip(debt / income, 0.0, 2.0) / 2.0),
            application.previous_repayment_rate,
            float(application.repeated_borrower),
            float(application.urgent_need),
            float(np.clip(application.dependants / 8.0, 0.0, 1.0)),
        ]
        if self.include_protected_attributes:
            values.insert(1, float(np.clip((application.age - 18) / 62.0, 0.0, 1.0)))
            values.extend([borrower.poverty_relief_score, borrower.vulnerability_score])
        values.extend(float(application.sector == value) for value in self.sectors)
        values.extend(
            float(application.loan_purpose == value) for value in self.purposes
        )
        values.extend(
            float(application.region_type == value) for value in self.region_types
        )
        if self.include_protected_attributes:
            values.extend(
                [
                    float(application.gender.value == "female"),
                    float(application.disability),
                    float(application.marginalized_community),
                    float(application.reserved_by_government),
                    float(application.age >= 60),
                ]
            )
        return np.asarray(values, dtype=float)

    def transform_many(self, borrowers: list[Borrower]) -> np.ndarray:
        if not borrowers:
            return np.empty((0, self.n_features), dtype=float)
        return np.vstack([self.transform(borrower) for borrower in borrowers])
