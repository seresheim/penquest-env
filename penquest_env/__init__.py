from gymnasium.envs.registration import register
from penquest_env.PenQuestEnv import PenQuestEnv
from penquest_env.network.connect import start

__version__ = "0.1.0"
__author__ = "Sebastian Eresheim, Alexander Piglmann, Simon Gmeiner, Thomas Peteling"
__credits__ = "PenQuest"

register(
    id="penquest_env/PenQuest-v0",
    entry_point="penquest_env:PenQuestEnv",
    reward_threshold=1.0,
    nondeterministic=True
)

__all__ = [start]