# PenQuest Environment

This repository contains the code for the Reinforcement Learning environment
for the game [PenQuest](https://www.pen.quest).

## Installation

To install this package please first install the package [penquest-pkgs](https://github.com/seresheim/penquest-pkgs).
Then install the this package, navigate into the folder and execute the following command:
```bash
    pip install -e .
```

Before execution please provide your API-Key via the environment variable
'API_KEY'. For your convenience this is best done via an '.env' file.

Due to limited computation ressources, please contact the authors for an 
API key to access the PenQuest API.

## Usage

The following is a minimal viable example usage of the environment for random 
actions:

```python
import random as rand
import os
import gymnasium as gym
import penquest_env
from penquest_env import Scenario, SlotType, PlayerType, BotType

api_key = os.getenv("API_KEY")
OPTIONS = {
    "scenario": Scenario.INFRASTRUCUTRE_SCENARIO_MEDIUM_1_ALL_ACTIONS,
    'slot': SlotType.ATTACK,
    'seed': 1234,
    'players': [
        { 'type': PlayerType.BOT, 'bot_type': BotType.ADVANCED_BOT },
    ],
}

if __name__ == "__main__":
    penquest_env.start(api_key)
    env = gym.make('penquest_env/PenQuest-v0', options=OPTIONS)
    obs, info = env.reset(options=OPTIONS)
    done = False
    while not done:
        action = rand.choice(info["valid_actions"])
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    env.close()
```