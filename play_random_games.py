import penquest_env
import random as rand
import gymnasium as gym


API_KEY = ""
OPTIONS = {
        "scenario": "92477f8d-d7fc-4cf2-b07c-e22bb12bebb4",
        'game_options': {
            'action_success_mode': 0, # 0 - off, 1 - always succeed
            'action_detection_mode': 0, # 0 - off, 1 - always detect
            'equipment_shop_mode': 1, # 0 - No shop, 1 - random shop, 2 - all items
            'action_shop_mode': 0, # 0 - random actions, 1 - all actions
            'support_actions_mode': 0, # 0 - no support actions, 1 - support actions
            'game_objectives_mode': 0, # 0 - default, 1 - random
            'initial_asset_stage': 0,  # 0 - default, 1 - rec, 2 - ini, 3 - exe
            'initial_action_mode': 1, # 0 - random, 1 - playable, 2 - pick
            'manual_def_type_mode': 0 # 0 - disabled, 1 - prevention, 2 - detection, 3 - response, 4 - prevention only, 5 - detection only, 6 - response only
        },
        'goal': 2,
        'slot': 1,
        'seed': 1234,
        'players': [
            { 'type': 'bot', 'bot_type': 1 },
        ],
    }

def play_single_random_game(env, steps=100):
    state, info = env.reset(options=OPTIONS)
    done = False
    step = 1
    while not done:
        print(f"Step: {step}")
        action = rand.choice(info["valid_actions"])
        print(f"Selected Action: {action}")
        next_obs, reward, terminated, truncated, info = env.step(action)
        step += 1
        done = terminated or truncated
    env.close()

if __name__ == "__main__":
    penquest_env.start(API_KEY)
    env = gym.make('penquest_env/PenQuest-v0', options=OPTIONS)
    for _ in range(2):
        play_single_random_game(env)
    print("Ended")