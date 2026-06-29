from dataclasses import dataclass


@dataclass
class LoanAction:
    """
    This is what the bandit chooses.
    the policy chooses among:

    reject
    approve with small amount
    approve with medium amount
    approve with large amount
    """

    action_id: str
    option_name: str
    approve: bool
    approved_loan_amount: float
    loan_term_steps: int
