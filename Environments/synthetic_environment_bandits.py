from __future__ import annotations

from collections.abc import Sequence

from Environments.base_environment import BanditEnvironment, BanditMode
from models.arm_spec import ArmSpec
from models.environment_config import EnvironmentConfig


class ArmBanditEnvironment(BanditEnvironment):
    """wrapper for the fixed-arm environment."""

    def __init__(self, arms: Sequence[ArmSpec], config: EnvironmentConfig) -> None:
        super().__init__(mode=BanditMode.ARM, arms=arms, config=config)


SyntheticEnvironment = ArmBanditEnvironment
