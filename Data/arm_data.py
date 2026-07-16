from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from models.arm_spec import ArmSpec


def default_arm_configuration() -> tuple[list[ArmSpec], dict[str, float]]:
    """Built-in distributions; standard deviation is 10% of mean cost."""

    return (
        [
            ArmSpec("Group A", "A", 0.80, cost_mean=12.0, cost_std=1.2),
            ArmSpec("Group B", "B", 0.65, cost_mean=8.0, cost_std=0.8),
            ArmSpec("Group C", "C", 0.50, cost_mean=4.0, cost_std=0.4),
        ],
        {"A": 0.20, "B": 0.30, "C": 0.30},
    )


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def load_arm_configuration(path: Path) -> tuple[list[ArmSpec], dict[str, float]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"arm configuration does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in arm configuration: {exc}") from exc

    root = _require_mapping(payload, "configuration")
    raw_arms = root.get("arms")
    if not isinstance(raw_arms, list) or not raw_arms:
        raise ValueError("configuration must contain at least one arm")
    required = {
        "name",
        "group",
        "repayment_probability",
        "cost_mean",
        "cost_std",
    }
    arms: list[ArmSpec] = []
    for index, raw_arm in enumerate(raw_arms):
        arm = _require_mapping(raw_arm, f"arms[{index}]")
        missing = required - set(arm)
        if missing:
            raise ValueError(
                f"arms[{index}] is missing: {', '.join(sorted(missing))}"
            )
        arms.append(
            ArmSpec(
                name=str(arm["name"]),
                group=str(arm["group"]),
                repayment_probability=float(arm["repayment_probability"]),
                cost_mean=float(arm["cost_mean"]),
                cost_std=float(arm["cost_std"]),
            )
        )

    names = [arm.name for arm in arms]
    if len(names) != len(set(names)):
        raise ValueError("arm names must be unique")
    groups = {arm.group for arm in arms}
    raw_targets = _require_mapping(root.get("target_shares", {}), "target_shares")
    targets = {str(group): float(value) for group, value in raw_targets.items()}
    unknown = set(targets) - groups
    if unknown:
        raise ValueError(
            "target_shares contains unknown groups: " + ", ".join(sorted(unknown))
        )
    if any(not 0.0 <= share <= 1.0 for share in targets.values()):
        raise ValueError("target shares must be in [0, 1]")
    if sum(targets.values()) > 1.0 + 1e-12:
        raise ValueError("target shares cannot sum to more than one")
    return arms, targets
