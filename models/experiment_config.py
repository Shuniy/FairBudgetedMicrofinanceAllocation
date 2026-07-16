from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Configuration parameters for the experiment."""
    seeds: int = 10
    output: Path = Path("outputs")
    save_curves: bool = True

    def __post_init__(self) -> None:
        if self.seeds <= 0:
            raise ValueError("seeds must be positive")
