from dataclasses import dataclass


@dataclass
class OutcomeModelParams:
    """
    This controls the simulated ground truth.
    """

    repayment_probability: float
    repayment_delay_steps: int
    expected_repayment_amount: float

    # Optional social outcomes
    realized_social_impact_mean: float
    realized_social_impact_std: float

    # Harm / over-indebtedness
    indebtedness_change_mean: float
    indebtedness_change_std: float
