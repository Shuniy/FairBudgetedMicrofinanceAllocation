from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from Helpers.enums import Gender, LoanPurpose, RegionType, Sector
from models.application_context import ApplicationContext
from models.borrower import Borrower
from phase2.feature_encoder import BorrowerFeatureEncoder
from phase2.models import BorrowerCandidate, ContextualEnvironmentConfig


@dataclass(slots=True)
class SyntheticBorrowerGenerator:
    config: ContextualEnvironmentConfig
    encoder: BorrowerFeatureEncoder
    rng: np.random.Generator
    applicant_counter: int = 0

    def _choice(self, values: tuple, probabilities: list[float] | None = None):
        return values[int(self.rng.choice(len(values), p=probabilities))]

    def _generate_application(self) -> ApplicationContext:
        self.applicant_counter += 1

        gender = (
            Gender.FEMALE
            if self.rng.random() < self.config.female_probability
            else Gender.MALE
        )
        marginalized = self.rng.random() < self.config.marginalized_probability
        reserved = self.rng.random() < self.config.reserved_probability
        disability = self.rng.random() < self.config.disability_probability

        region_type = self._choice(
            tuple(RegionType),
            [
                1.0 - self.config.rural_probability - 0.15,
                0.15,
                self.config.rural_probability * 0.75,
                self.config.rural_probability * 0.25,
            ],
        )
        sector = self._choice(tuple(Sector))
        purpose = self._choice(tuple(LoanPurpose))

        age = int(np.clip(self.rng.normal(39, 13), 18, 80))
        base_income = self.rng.lognormal(mean=np.log(35_000), sigma=0.65)
        if region_type in {RegionType.RURAL, RegionType.ISOLATED}:
            base_income *= 0.72
        if marginalized or reserved:
            base_income *= 0.82
        income = float(np.clip(base_income, 4_000, 220_000))

        requested_fraction = self.rng.uniform(0.08, 0.45)
        requested_loan = float(np.clip(income * requested_fraction, 1_000, 40_000))
        existing_debt = float(np.clip(income * self.rng.beta(1.6, 5.0), 0, 100_000))
        repeated = bool(self.rng.random() < 0.35)
        urgent = bool(self.rng.random() < 0.18)
        previous_rate = float(
            np.clip(
                self.rng.beta(8, 2) if repeated else self.rng.beta(5, 3),
                0.05,
                0.99,
            )
        )

        return ApplicationContext(
            applicant_id=f"APP-{self.applicant_counter:07d}",
            name=f"Borrower {self.applicant_counter}",
            age=age,
            gender=gender,
            income=income,
            region_name=f"Region-{int(self.rng.integers(1, 21))}",
            region="synthetic",
            loan_purpose=purpose,
            sector=sector,
            region_type=region_type,
            repeated_borrower=repeated,
            urgent_need=urgent,
            requested_loan_amount=requested_loan,
            existing_debt=existing_debt,
            previous_repayment_rate=previous_rate,
            dependants=int(self.rng.integers(0, 7)),
            disability=disability,
            marginalized_community=marginalized,
            reserved_by_government=reserved,
        )

    @staticmethod
    def true_repayment_probability(borrower: Borrower) -> float:
        """Hidden simulator rule; policies never observe this probability."""

        app = borrower.application_context
        income_strength = np.clip(np.log1p(app.income) / np.log1p(200_000), 0, 1)
        loan_to_income = np.clip(app.requested_loan_amount / app.income, 0, 2)
        debt_to_income = np.clip(app.existing_debt / app.income, 0, 2)

        sector_effect = {
            Sector.AGRICULTURE: -0.20,
            Sector.RETAIL: 0.05,
            Sector.SERVICES: 0.15,
            Sector.MANUFACTURING: 0.08,
            Sector.OTHER: 0.0,
        }[app.sector]
        purpose_effect = {
            LoanPurpose.BUSINESS: 0.15,
            LoanPurpose.AGRICULTURE: -0.10,
            LoanPurpose.EDUCATION: 0.08,
            LoanPurpose.HEALTH: -0.18,
            LoanPurpose.HOUSING: 0.02,
            LoanPurpose.OTHER: 0.0,
        }[app.loan_purpose]

        # Protected attributes are deliberately not direct causal terms.
        # Correlated financial/geographic features can still create disparities.
        logit = (
            -0.65
            + 1.20 * income_strength
            + 2.10 * app.previous_repayment_rate
            + 0.30 * float(app.repeated_borrower)
            - 1.55 * loan_to_income
            - 1.15 * debt_to_income
            - 0.35 * float(app.urgent_need)
            - 0.25 * float(borrower.geographically_isolated)
            + sector_effect
            + purpose_effect
        )
        probability = 1.0 / (1.0 + np.exp(-logit))
        return float(np.clip(probability, 0.05, 0.95))

    def generate_candidate(self) -> BorrowerCandidate:
        borrower = Borrower(self._generate_application())
        context = self.encoder.transform(borrower)
        probability = self.true_repayment_probability(borrower)
        return BorrowerCandidate(
            borrower=borrower,
            context=context,
            true_repayment_probability=probability,
        )

    def generate_pool(self, pool_size: int) -> list[BorrowerCandidate]:
        return [self.generate_candidate() for _ in range(pool_size)]
