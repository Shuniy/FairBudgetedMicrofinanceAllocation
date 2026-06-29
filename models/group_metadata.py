from dataclasses import dataclass


@dataclass
class GroupMetadata:
    """
    This lets define policy-level group rules without repeating them on every applicant.
    """

    group_name: str
    mandatory_service: bool
    policy_priority_weight: float
    minimum_allocation_share: float
    minimum_budget_share: float
