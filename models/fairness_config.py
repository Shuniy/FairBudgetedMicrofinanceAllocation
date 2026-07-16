from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Mapping


@dataclass(frozen=True, slots=True)
class FairnessConfig:
    """Configuration parameters for fairness constraints."""
    target_shares: Mapping[str, float] = field(default_factory=dict)
    demographic_parity_tolerance: float = 1.0
    penalty_lambda: float = 0.0

    def __post_init__(self) -> None:
        if any(not 0.0 <= share <= 1.0 for share in self.target_shares.values()):
            raise ValueError("target shares must be in [0, 1]")
        if sum(self.target_shares.values()) > 1.0 + 1e-12:
            raise ValueError("target shares cannot sum to more than one")
        if not 0.0 <= self.demographic_parity_tolerance <= 1.0:
            raise ValueError("demographic_parity_tolerance must be in [0, 1]")
        if self.penalty_lambda < 0:
            raise ValueError("penalty_lambda cannot be negative")
