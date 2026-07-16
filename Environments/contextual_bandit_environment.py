from __future__ import annotations

from Data.feature_encoder import BorrowerFeatureEncoder
from Environments.base_environment import BanditEnvironment, BanditMode
from models.environment_config import ContextualEnvironmentConfig


class ContextualBanditEnvironment(BanditEnvironment):
    """wrapper for the contextual environment."""

    def __init__(
        self,
        config: ContextualEnvironmentConfig,
        encoder: BorrowerFeatureEncoder | None = None,
    ) -> None:
        super().__init__(mode=BanditMode.CONTEXTUAL, config=config, encoder=encoder)


ContextualMicrofinanceEnv = ContextualBanditEnvironment
