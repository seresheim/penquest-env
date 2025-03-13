import random as rand
import os

import gymnasium as gym

from penquest_env import (
    GameOptions,
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
    Scenario,
    SlotType,
    PlayerType,
    BotType,
)
import penquest_env


api_key = os.getenv("API_KEY")
game_options = GameOptions(
    action_success_mode=ActionSuccessMode.DEFAULT,
    action_detection_mode=ActionDetectionMode.DEFAULT,
    equipment_shop_mode=EquipmentShopMode.ALL_EQUIPMENT,
    action_shop_mode=ActionShopMode.ALL_ACTIONS,
    support_actions_mode=SupportActionsMode.ENABLED,
    game_objectives_mode=GameObjectivesMode.DEFAULT,
    initial_asset_stage=InitialAssetStage.DEFAULT,
    initial_action_mode=InitActionsMode.PLAYABLE,
    manual_def_type_mode=ManualDefType.DISABLED,
    infinite_shields=False,
    multi_target_success=MultiTargetSuccess.ONE_PER_TARGET,
    defender_actions_detectable=DefenderActionsDetectable.RESPONSE_ONLY,
    availability_penalty=DefenderAvailibilityPenalty.ENABLED,
    defender_pre_setup_mode=DefenderPreSetupMode.ATTRIBUTE_BASED,
)
OPTIONS = {
    "scenario": Scenario.INFRASTRUCUTRE_SCENARIO_MEDIUM_1_ALL_ACTIONS,
    'slot': SlotType.ATTACK,
    'game_options': game_options,
    #'goal': 2,
    'seed': 1234,
    'players': [
        { 'type': PlayerType.BOT, 'bot_type': BotType.ADVANCED_BOT },
    ],
}

def play_parallel_random_games(env, steps=100):
    state, info = env.reset(options=OPTIONS)
    step = 1
    for _ in range(steps):
        print(f"Step: {step}")
        action = tuple([rand.choice(valid_actions) for valid_actions in info["valid_actions"]])
        print(f"Selected Action: {action}")
        next_obs, reward, terminated, truncated, info = env.step(action)
        step += 1
    env.close()

if __name__ == "__main__":
    penquest_env.start(api_key)
    env = gym.make_vec(
        'penquest_env/PenQuest-v0', 
        num_envs=3,
        #vector_kwargs={'shared_memory':False},
        options=OPTIONS
    )
    rand.seed(OPTIONS['seed'])
    play_parallel_random_games(env)
    print("Ended")