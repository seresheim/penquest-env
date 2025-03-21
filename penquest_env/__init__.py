from gymnasium.envs.registration import register

from penquest_pkgs.model import GameOptionsModel as GameOptions
from penquest_pkgs.constants import (
    ActionDetectionMode,
    ActionShopMode,
    ActionSuccessMode,
    DefenderActionsDetectable,
    DefenderAvailibilityPenalty,
    DefenderPreSetupMode,
    EquipmentShopMode,
    GameObjectivesMode,
    InitActionsMode,
    InitialAssetStage,
    ManualDefType,
    MultiTargetSuccess,
    SupportActionsMode,
)

from penquest_env.constants import Scenario, SlotType, PlayerType, BotType
from penquest_env.PenQuestEnv import PenQuestEnv
from penquest_env.network.connect import start

__version__ = "0.2.2"
__author__ = "Sebastian Eresheim, Alexander Piglmann, Simon Gmeiner, Thomas Petelin"
__credits__ = "PenQuest"

register(
    id="penquest_env/PenQuest-v0",
    entry_point="penquest_env:PenQuestEnv",
    reward_threshold=1.0,
    nondeterministic=True
)

__all__ = [
    "PenQuestEnv",
    "start",
    "GameOptions",
    "ActionDetectionMode",
    "ActionShopMode",
    "ActionSuccessMode",
    "DefenderActionsDetectable",
    "DefenderAvailibilityPenalty",
    "DefenderPreSetupMode",
    "EquipmentShopMode",
    "GameObjectivesMode",
    "InitActionsMode",
    "InitialAssetStage",
    "ManualDefType",
    "MultiTargetSuccess",
    "SupportActionsMode",
    "Scenario",
    "SlotType",
    "PlayerType",
    "BotType",
]