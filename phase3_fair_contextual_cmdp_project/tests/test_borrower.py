from Helpers.enums import Gender, LoanPurpose, RegionType, Sector
from models.application_context import ApplicationContext
from models.borrower import Borrower


def test_borrower_derived_features_are_initialized() -> None:
    borrower = Borrower(
        ApplicationContext(
            applicant_id="A-1",
            name="Example",
            age=35,
            gender=Gender.FEMALE,
            income=18_000,
            region_name="Rural-1",
            region="synthetic",
            loan_purpose=LoanPurpose.BUSINESS,
            sector=Sector.RETAIL,
            region_type=RegionType.RURAL,
            repeated_borrower=True,
            urgent_need=False,
            requested_loan_amount=4_000,
            existing_debt=1_000,
            previous_repayment_rate=0.9,
            marginalized_community=True,
        )
    )

    assert borrower.protected_groups
    assert borrower.socially_marginalized
    assert borrower.geographically_isolated
    assert 0.0 <= borrower.poverty_relief_score <= 1.0
