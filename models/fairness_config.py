from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Mapping


def _validate_target_shares(target_shares: Mapping[str, float]) -> None:
    if any(not 0.0 <= share <= 1.0 for share in target_shares.values()):
        raise ValueError("target shares must be in [0, 1]")
    if sum(target_shares.values()) > 1.0 + 1e-12:
        raise ValueError("target shares cannot sum to more than one")


@dataclass(frozen=True, slots=True)
class FairnessConfig:
    """Configuration parameters for fairness constraints."""
    target_shares: Mapping[str, float] = field(default_factory=dict)
    demographic_parity_tolerance: float = 1.0
    penalty_lambda: float = 0.0

    def __post_init__(self) -> None:
        _validate_target_shares(self.target_shares)
        if not 0.0 <= self.demographic_parity_tolerance <= 1.0:
            raise ValueError("demographic_parity_tolerance must be in [0, 1]")
        if self.penalty_lambda < 0:
            raise ValueError("penalty_lambda cannot be negative")


@dataclass(frozen=True, slots=True)
class LagrangianConfig:
    """Dual-update parameters for Phase 3's soft fairness constraints."""

    target_shares: Mapping[str, float] = field(default_factory=dict)
    parity_tolerance: float = 0.05
    quota_tolerance: float = 0.01
    dual_learning_rate: float = 0.20
    initial_lambda_quota: float = 0.20
    initial_lambda_parity: float = 0.20
    max_lambda: float = 8.0
    warmup_rounds: int = 5

    def __post_init__(self) -> None:
        _validate_target_shares(self.target_shares)
        if not 0.0 <= self.parity_tolerance <= 1.0:
            raise ValueError("parity_tolerance must be in [0, 1]")
        if self.quota_tolerance < 0:
            raise ValueError("quota_tolerance cannot be negative")
        if self.dual_learning_rate <= 0:
            raise ValueError("dual_learning_rate must be positive")
        if self.initial_lambda_quota < 0 or self.initial_lambda_parity < 0:
            raise ValueError("initial Lagrange multipliers cannot be negative")
        if self.max_lambda <= 0:
            raise ValueError("max_lambda must be positive")
        if max(self.initial_lambda_quota, self.initial_lambda_parity) > self.max_lambda:
            raise ValueError("initial Lagrange multipliers cannot exceed max_lambda")
        if self.warmup_rounds < 0:
            raise ValueError("warmup_rounds cannot be negative")
