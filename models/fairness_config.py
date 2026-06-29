from dataclasses import dataclass, field


@dataclass
class FairnessConfig:
    """
    target_shares by regions
    quotas
    demographic_parity_tolerance
    equal_opportunity_tolerance
    mandatory group to provide loan or not
    minimum allocation to that group
    """

    # Hard fairness constraints
    min_allocation_share_by_group: dict[str, float] = field(default_factory=dict)
    min_budget_share_by_group: dict[str, float] = field(default_factory=dict)
    target_share_by_region: dict[str, float] = field(default_factory=dict)

    # Group protection / mandatory service
    must_serve_groups: set[str] = field(default_factory=set)

    # Tolerances
    demographic_parity_tolerance: float = 0.1
    equal_opportunity_tolerance: float = 0.1

    # Optional knobs to switch fairness definitions on/off
    use_quota_fairness: bool = True
    use_demographic_parity: bool = True
    use_equal_opportunity_proxy: bool = False
    use_budget_share_fairness: bool = False
