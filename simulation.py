from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from Data.borrower import Borrower
from Data.feature_encoder import BorrowerFeatureEncoder
from Helpers.enums import Gender, LoanPurpose, RegionType, Sector
from models.application_context import ApplicationContext
from models.borrower_candidate import BorrowerCandidate
from models.environment_config import ContextualEnvironmentConfig


def environment_rngs(seed: int) -> tuple[np.random.Generator, np.random.Generator]:
    """Create the opportunity and potential-outcome streams for an episode."""
    opportunity_seed, outcome_seed = np.random.SeedSequence(seed).spawn(2)
    return np.random.default_rng(opportunity_seed), np.random.default_rng(outcome_seed)


def policy_rng(seed: int) -> np.random.Generator:
    """Use the third child of the experiment seed for policy randomness."""
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(2,)))


@dataclass(slots=True)
class SyntheticBorrowerGenerator:
    """Generates synthetic borrower candidates for simulation."""
    config: ContextualEnvironmentConfig
    encoder: BorrowerFeatureEncoder
    rng: np.random.Generator
    applicant_counter: int = 0

    def _choice(self, values: tuple, probabilities: list[float] | None = None):
        """Randomly select a value from the given values, optionally using specified probabilities."""
        return values[int(self.rng.choice(len(values), p=probabilities))]

    def _generate_application(self) -> ApplicationContext:
        """Generate a synthetic application context for a borrower."""
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
        # Phases 2 and 3 use fixed loan bounds regardless of minimum_cost.
        requested_loan = float(
            np.clip(income * self.rng.uniform(0.08, 0.45), 1_000, 40_000)
        )
        existing_debt = float(
            np.clip(income * self.rng.beta(1.6, 5.0), 0, 100_000)
        )
        repeated = bool(self.rng.random() < 0.35)
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
            urgent_need=bool(self.rng.random() < 0.18),
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
        """Hidden outcome rule, unavailable to learning policies."""

        application = borrower.application_context
        income_strength = np.clip(
            np.log1p(application.income) / np.log1p(200_000), 0, 1
        )
        loan_to_income = np.clip(
            application.requested_loan_amount / application.income, 0, 2
        )
        debt_to_income = np.clip(
            application.existing_debt / application.income, 0, 2
        )
        sector_effect = {
            Sector.AGRICULTURE: -0.20,
            Sector.RETAIL: 0.05,
            Sector.SERVICES: 0.15,
            Sector.MANUFACTURING: 0.08,
            Sector.OTHER: 0.0,
        }[application.sector]
        purpose_effect = {
            LoanPurpose.BUSINESS: 0.15,
            LoanPurpose.AGRICULTURE: -0.10,
            LoanPurpose.EDUCATION: 0.08,
            LoanPurpose.HEALTH: -0.18,
            LoanPurpose.HOUSING: 0.02,
            LoanPurpose.OTHER: 0.0,
        }[application.loan_purpose]
        logit = (
            -0.65
            + 1.20 * income_strength
            + 2.10 * application.previous_repayment_rate
            + 0.30 * float(application.repeated_borrower)
            - 1.55 * loan_to_income
            - 1.15 * debt_to_income
            - 0.35 * float(application.urgent_need)
            - 0.25 * float(borrower.geographically_isolated)
            + sector_effect
            + purpose_effect
        )
        probability = 1.0 / (1.0 + np.exp(-logit))
        return float(np.clip(probability, 0.05, 0.95))

    def generate_candidate(self) -> BorrowerCandidate:
        borrower = Borrower(self._generate_application())
        return BorrowerCandidate(
            borrower=borrower,
            context=self.encoder.transform(borrower),
            true_repayment_probability=self.true_repayment_probability(borrower),
        )

    def generate_pool(self, pool_size: int) -> list[BorrowerCandidate]:
        return [self.generate_candidate() for _ in range(pool_size)]
