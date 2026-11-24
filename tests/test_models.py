from __future__ import annotations

import numpy as np
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
    borrower = Borrower(application(disability=True, age=65, reserved_by_government=True))
    control = Borrower(application(gender=Gender.MALE, marginalized_community=False))
    safe = BorrowerFeatureEncoder()
    audited = BorrowerFeatureEncoder(include_protected_attributes=True)
    assert safe.n_features == 25
    assert audited.n_features == 33
    assert borrower.poverty_relief_score != control.poverty_relief_score
    assert borrower.vulnerability_score != control.vulnerability_score
    np.testing.assert_array_equal(safe.transform(borrower), safe.transform(control))
    assert safe.transform_many([borrower, control]).shape == (2, 25)
    assert audited.transform_many([borrower, control]).shape == (2, 33)
    values = dict(zip(audited.feature_names, audited.transform(borrower)))
    assert values["age_scaled"] == pytest.approx(47 / 62)
    assert values["poverty_relief_score"] == borrower.poverty_relief_score
    assert values["vulnerability_score"] == borrower.vulnerability_score
    for feature in ("gender_female", "disability", "marginalized", "reserved", "aged"):
        assert values[feature] == 1
    for feature in (
        "age_scaled", "poverty_relief_score", "vulnerability_score",
        "gender_female", "disability", "marginalized", "reserved", "aged",
    ):
        assert feature not in safe.feature_names
