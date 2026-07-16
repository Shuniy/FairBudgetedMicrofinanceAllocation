from __future__ import annotations

import pytest

from Data.borrower import Borrower
from Data.feature_encoder import BorrowerFeatureEncoder
from Helpers.enums import Gender, IncomeGroup, LoanPurpose, RegionType, Sector
from models.application_context import ApplicationContext
from models.arm_spec import ArmSpec
from models.environment_config import ContextualEnvironmentConfig


def application(**overrides) -> ApplicationContext:
    values = {
        "applicant_id": "A-1",
        "name": "Example",
        "age": 35,
        "gender": Gender.FEMALE,
        "income": 18_000.0,
        "region_name": "Rural-1",
        "region": "synthetic",
        "loan_purpose": LoanPurpose.BUSINESS,
        "sector": Sector.RETAIL,
        "region_type": RegionType.RURAL,
        "repeated_borrower": True,
        "urgent_need": False,
        "requested_loan_amount": 4_000.0,
        "existing_debt": 1_000.0,
        "previous_repayment_rate": 0.9,
        "dependants": 2,
        "marginalized_community": True,
    }
    values.update(overrides)
    return ApplicationContext(**values)


def test_application_and_arm_validation() -> None:
    with pytest.raises(ValueError, match="repayment"):
        application(previous_repayment_rate=1.1)
    with pytest.raises(ValueError, match="cost_mean"):
        ArmSpec("bad", "A", 0.5, 0.0)
    with pytest.raises(ValueError, match="rural_probability"):
        ContextualEnvironmentConfig(
            total_budget=100,
            max_rounds=2,
            rural_probability=0.90,
        )


def test_borrower_derives_features_and_repayment_history() -> None:
    borrower = Borrower(application())
    assert borrower.previous_repayment_rate == 0.9
    assert borrower.socially_marginalized
    assert borrower.geographically_isolated
    assert 0.0 <= borrower.poverty_relief_score <= 1.0
    assert 0.0 <= borrower.vulnerability_score <= 1.0


def test_upper_middle_income_band_is_reachable() -> None:
    assert Borrower(application(income=100_000)).income_group == IncomeGroup.UPPER_MIDDLE


def test_protected_features_are_excluded_by_default() -> None:
    borrower = Borrower(application(disability=True))
    safe = BorrowerFeatureEncoder()
    audited = BorrowerFeatureEncoder(include_protected_attributes=True)
    assert audited.n_features == safe.n_features + 5
    assert safe.transform(borrower).shape == (safe.n_features,)
    assert not any("female" in name for name in safe.feature_names)
