Getting Started
===============

The following is a code snippet of a minimum example to run PenQuest-Env:

.. code-block::

    import penquest_env
    import gymnasium as gym
    import random as rand

    OPTIONS = {
            "scenario": 9,
            'players': [
                { 'type': 'bot', 'bot_type': 1 },
            ],
        }

    penquest_env.start(api_key="123456789", host="ws://env.pen.quest", port=5000)
    env = gym.make('penquest_env/PenQuest-v0', options=OPTIONS)

    state, info = env.reset(options=OPTIONS)
    done = False
    while not done:
        action = rand.choice(info["valid_actions"])
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    env.close()

PenQuest-Env requires two imports, `penquest_env` to load the actual 
PenQuest package and `gymnasium <https://gymnasium.farama.org/>`_ to make the 
environment and for additional potential wrappers. PenQuest-Env supports 
gymnasium v26. Before you can instantiate an environment via gymnasium, you 
first need to start the PenQuest-Env package using `penquest_env.start()`. 
PenQuest-Env uses a separate process for communicating with the PenQuest Server, 
in order to channel the communication of potential many virutal environments 
over a single websocket connection. By calling `penquest_env.start()` this 
communication process is created and starts listening for local environments.
The `penquest_env.start()` receives parameters to know where to connect to and
to identify the client by using an API-key. You can get an API-key by getting in
contact with the PenQuest developers at https://www.pen.quest/. 
PenQuest-Env also supports configuration files for storing configuration data
like host adresses, ports, API-keys etc. For more information on this see 
:doc:`Configuration Files <config_files>`.

.. _penquest_env.start: 
.. autofunction:: penquest_env.start

After the package was started, you can create a PenQuest environment likey any
other gymnasium environment using `gym.make 
<https://gymnasium.farama.org/api/registry/#gymnasium.make>`_ via the 
*penquest_env/PenQuest-v0* identifier. 

Options
-------

Options configure the game the agent is playing in and are passed via a 
dictionary as a key word parameter to the `reset() 
<https://gymnasium.farama.org/api/env/#gymnasium.Env.reset>`_ method of the 
environment. Valid fields in the dictionary are:

:**scenario**: set the scenario of the PenQuest game by providing the ID of the 
  scenario. For a full list of scenario IDs see ...
:**goal**: sets the goal for the attacker of the game in case the scenario has
  multiple goals defined. In case there are multiple goals defined, usually the
  last option (-1) is a random pick from all the defined goals.
:**game options**: game internal settings that change the gameplay. For more
  information sett :doc:`Game Options <game_options>`
:**players**: a list of dictionaries that determines the other players in the 
    game. Other players can either be of *type* `bot` or `human`. Bots have an
    additional *bot_type*, either 0 for a random bot or 1 for a more intelligent
    bot. 
:**slot**: determines which slot in the scenario the agent will be in. Most of
    the scenarios are currently two-player scenarios where the attacker is on
    slot 1 and the defender on slot 2, but this highly depends on the specific 
    scenario.
:**join**: joins an already opened game with a human player. Value is the game
    key to join. 
:**wait_for_players**: specifies the amount of seconds the environment waits for
    a human player to join a game until it stops waiting.
:**seed**: seeds the random number generator of the environment. This is useful
    for reproducibility of experiments.

