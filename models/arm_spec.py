from dataclasses import dataclass

# All plausible fairness constraints
# Quota-based fairness - Each protected group or region must receive at least a minimum allocation share.
# Example: rural borrowers must get at least 20%, geographically isolated regions must get at least 15%
#
# Demographic parity / allocation parity - Allocation rates across groups should not diverge beyond a tolerance.
# Example - approval rate for group A and group B should differ by no more than delta
#
# Equal opportunity - Among “qualified” or “repayment-capable” applicants, allocation rates should be similar across groups.
# Example - among applicants with high repayment probability, approval rates for group A and group B should differ by no more than delta
# This is harder because we need a notion of merit or repayment-capable ground truth.
#
# Minimum service / mandatory coverage constraint - Some groups must never be completely excluded.
# Example: every protected region must receive at least k loans over a horizon or at least one feasible loan every H rounds
#
#
# Budget-share fairness - A group should receive at least a minimum share of capital, not just number of approvals.
# Example: rural borrowers must receive at least 25% of the total allocated capital, even if their approval rate is lower than urban borrowers.
# This matters because counting approvals alone can be misleading if one group only gets very tiny loans.
#
# Opportunity-cost fairness The system should not systematically allocate only low-value / low-opportunity loans to disadvantaged groups.
# Example: the average loan amount for rural borrowers should not be less than 50% of the average loan amount for urban borrowers.
# This is not as easy to implement at the start, but it is a good analysis metric later.
#
#
# Fairness-as-penalty - Instead of hard constraints, we could add a penalty term to the reward function that penalizes deviations from fairness criteria.
# Example: the reward for an allocation could be adjusted by subtracting a penalty proportional to the difference in approval rates between groups.
# Not a constraint itself, but a way to encode fairness.
# The algorithm loses score when it under-allocates to protected groups. Example: score = utility score − lambda * fairness_violation
#
#
# Fairness-as-cost / knapsack-style cost - Fairness violation is treated like another resource cost to control.
#
# Region target-share constraint - Each region gets a target share, based on policy or mission.
#
# Need-weighted fairness - More disadvantaged groups may deserve higher priority because equality is not always equity.
#
#


@dataclass
class ArmSpec:
    name: str
    group: str
    repayment_probability: float
    cost_mean: float
    cost_std: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.repayment_probability <= 1.0:
            raise ValueError("repayment_probability must be in [0, 1]")
        if self.cost_mean <= 0:
            raise ValueError("cost_mean must be strictly positive")
        if self.cost_std < 0:
            raise ValueError("cost_std cannot be negative")


@dataclass
class ContextualArmSpec:
    name: str

    # Group / region descriptors
    protected_group: str
    region: str
    region_type: str
    income_group: str
    socially_marginalized: bool
    geographically_isolated: bool

    # Policy / importance descriptors
    priority_weight: float
    mandatory_service_group: bool

    # Cost / loan profile
    expected_cost: float
    cost_std: float

    # Outcome profile
    repayment_probability: float
    repayment_delay_steps: int
    expected_repayment_amount: float

    # Social-good dimensions
    poverty_relief_score: float
    social_impact_score: float
    indebtedness_risk_score: float
