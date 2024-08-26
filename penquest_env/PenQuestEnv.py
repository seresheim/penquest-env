import os
import asyncio
import gymnasium as gym
import numpy as np

from itertools import permutations
from typing import Dict, Optional, Tuple, Set, List, Any
from gymnasium.spaces import (
    Discrete,
    Sequence,
    Text,
    Box,
    MultiDiscrete,
    MultiBinary
)
from penquest_env.ConnectionHelper import ConnectionHelper
from penquest_env.ObservationFactory import ObservationFactory

from penquest_pkgs.constants import GameEndedState, GameInteractionType
from penquest_pkgs.game import Game
from penquest_pkgs.utils import get_logger


HUMAN_PLAYER = 'human'
DEFAULT_SCENARIO = 9

class GameConfig():
    PLAYER_ID = "player_id"
    LOBBY = "lobby"
    JOIN = "join"
    SCENARIO_ID = "scenario_id"
    OPTIONS = "options"
    SLOT = "slot"
    PLAYERS = "players"
    TYPE = "type"
    BOT_TYPE = "bot_type"
    WAIT_FOR_PLAYERS = "wait_for_players"


DEFAULT_WAIT_FOR_PLAYERS_PERIOD = 240
MAX_AMOUNT_OF_TURNS = int(1e10)
MAX_AMOUNT_OF_ACTORS_IN_GAME = 64
MAX_AMOUNT_SELECTION = 20
MAX_AMOUNT_OF_ASSETS = int(1e10)
MAX_AMOUNT_OF_ACTIONS = int(1e10)
MAX_AMOUNT_OF_ACTION_TEMPLATES = int(1e10)
MAX_AMOUNT_OF_EQUIPMENT_TEMPLATES = int(1e10)
MAX_AMOUNT_OF_ASSET_CATEGORIES = 20
MAX_EFFECT_CATEGORIES = 100000
AMOUNT_OF_EFFECT_TYPES = 10
MAX_AMOUNT_OF_GOAL_TYPES = 4
MAX_AMOUNT_OF_OSES = 7
MAX_AMOUNT_OF_ATTACK_STAGES = 4

ATTACK_MASKS = [
    "",
    "C",
    "I",
    "A",
    "CI",
    "IA",
    "CA",
    "CIA"
]

CHARSET ="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789,;.:-_^°!\"§$%&/()=?\\'# *~<>| "

KEY_INITIAL_ACTION_MODE = "initial_action_mode"
KEY_EQUIPMENT_SHOP_MODE = "equipment_shop_mode"

DEFAULT_CONFIG_FILE = "default_config.ini"


class PenQuestEnv(gym.Env):
    """
    Description
    -----------

    For a brief description of the game PenQuest, please see the official 
    documentation at https://penquest-env.readthedocs.com/ or the website at
    https://www.pen.quest/ .

    Action Space
    ------------

    The action space is a sequence of discrete spaces with values ranging from 
    0 to 1e10-1. Examples of actions are:

      - `(3, 0, 1, 2, 4)`
      - `(3, 0, 0, 0, 4, 0)`
      - `(6,)`
      - `()`
    
    Action tuples have different meanings depending on the current phase of the
    game. In general the agent needs to interact in three separate ways (in that
    specific order for each turn):

      1. Buy equipment from the shop
      2. Redraw an action card from the stack. The agent can choose -- depending on the game option -- from a pre-selection or the entire deck. This step is omitted in the first turn.
      3. Play an action card. 

    Shop and Redraw
    ...............

    In cases 1 and 2, the integers in the action tuple resemble indices of lists.
    For the first case this list is the sortiment of the shop where at each
    position a different equipment resides. In the second case, the list is a 
    list of possible actions to redraw from the deck to the agent's hand.
    Depending on the game options and the scenario, this list might either be 
    a pre-selected subset of the whole action deck or the whole action deck
    itself. Empty tuples indicate, that no equipment was bought from the shop (
    not redrawing an action card when one was played before is not a valid 
    option, however shopping is completely optional).

    .. note::

        Because different scenarios might include different actions/equipment,
        the total amount of actions/equipment might vary in sice largly. This
        is the reason why the upper limit of the discrete space is set to 1e10. 
        In practice it is highly unlikely that any scenario includes more than 
        that amount of actions/equipment.

    .. admonition:: example: buy
    
        The action `(3, 0, 1, 2, 4)` translates to buying the 4th, 1st, 2nd, 
        3rd and 5th equipment of the shop.

    .. admonition:: example: redraw
    
        The action `(3, 6)` translates to redraw the 4th and 7th action from the
        pre-selection to the agent's hand.

    Play Action
    ...........

    In case 3 the size of the tuple is always 6 and each position in the tuple
    has it's own meaning:

    :Position 0: Index (starting at 0) of the action card of the agent's hand 
        that is played. This field is required!
    :Position 1: ID of the asset the action card is played onto. This field is
        optional, 0 indicates no asset selected.
    :Position 2: Indicates the attack mask that is used to play action with.
        This field is an index (starting with 0) into a list that consists of 
        the values: `["", "C", "I", "A", "CI", "IA", "CA", "CIA"]`
    :Position 3: Index (starting at 1) of a support action card of the agent's 
        hand that is played along with the main action at position 0. This field
        is optional, 0 indicates no support action was chosen.
    :Position 4: Index (starting at 1) of an equipment card of the agent that is
        played along with the main action at position 0. This field is optional,
        0 indicates no equipment card was chosen.
    :Position 5: ID of a main action card, which was previously played onto an 
        asset and the current main action card at position 0 is supposed to
        counter. This field can only be set if the field at position 1 is also
        set (> 0). This field is optional, 0 indicates no response target was
        chosen.

    .. admonition:: example: play action
    
        The action `(3, 15, 2, 5, 4, 0)` translates to play action at position 3
        of the agent's hand onto asset with ID 15 using attack mask "I",
        supporting with action at position 5 of the agent's hand and equipt with
        the equipment at position 4 of the agent's hand. No response target was
        selected.

    Observation Space
    -----------------
      
    The observation space consists of a dict space that contains information
    about the same objects a human receives via the web interface. The following
    is a listing of key-value pairs of the top-level observation space, where 
    each value is another type of gym space:

    :`board`: Sequence(Dict(...)) - each dict in the sequence models a single 
        asset.
    :`hand`: Sequence(Dict(...))- each dict in the sequence models a single 
        action card.
    :`equipment`: Sequence(Dict(...)) - each dict in the sequence models a
        single equipment action card.
    :`shop`: Sequence(Dict(...)) - each dict in the sequence models a single 
        equipment action card.
    :`roles`: Sequence(Dict(...)) - each dict in the sequence models a single 
        actor role.
    :`actor_id`: Discrete(64) - ID of a player in the game.
    :`actor_connection_id`: Text(max_len=50) - unique identifier for the
        connection to the game (not the websocket connection!) This is 
        necessary, because the actors can play against themselves, thus the
        actor_id alone is not enough.
    :`turn`: Discrete(1e10) - indicates the current turn in the game.
    :`phase`: Discrete(6) - indicates the current phase of the game.
    :`selection_choices`: Sequence(Dict(...)) - each dict in the sequence 
        models a single action card, representing choices available to the agent
        for redrawing.
    :`selection_amount`: Discrete(20) - indicates the number of selections
        required in the current turn.

    In the following subsections all sub-spaces of the observation space are
    listed in detail.

    Asset Space
    ...........

    Assets are represented via a dictionary (space) and mostly in the `'board'`
    field of the general observation space. 

    :`id`: Discrete(1e10) - ID of the asset.
    :`category`: Discrete(20) - each asset has one of the following categories:

        :1: Mobile - models a mobile phone device (e.g. Android phone, iPhone, 
            etc.)
        :2: Cloud - any cloud device (e.g. AWS instances)
        :3: DMZ (de-militarized zone) - network infrastructure that usually 
            contains assets which are reachable from the internet 
        :4: LAN - common network infrastrucutre to segregate network parts
        :5: Web Server - models any type of web server (e.g.: apache http server,
             IIS, Nginx, etc.)
        :6: Mail Server - models any type of e-mail server (e.g. Microsoft 
            Exchange, OpenSMTPD, etc.)
        :7: App Server - models anny kind of application server. These can also
            be self-coded services of a company.
        :8: File Server - models any kind of file server (e.g. SMB, NFS, etc.)
        :9: Database - models any kind of database (e.g. PostgreSQL, Oracle, 
            Microsoft SQL server, etc.)
        :10: Client - models any kind of enpoint system, like a normal work
            station
        :11: Network Appliance - models any kind of uncommon network 
            infrastructure
        :12: IoT - models any kind of IoT device (e.g. smart home devices)
        :13: Industrial System - models any kind of OT device
        :14: SCADA - models any kind of SCADA device
        :15: Device - models any device that does not fit a previous category

        Integers that are not listed, are currently unused, but reserved for
        future use.

    :`os`: Discrete(5) - integer that indicates which operating system the 
        asset is running. Possible values are:

        :1: Windows
        :2: Linux
        :3: iOS
        :4: Android
        :5: Cloud
        :6: MacOS

    :`attack_stage`: Discrete(4) - integer that indicates which attack stage the
        attacker is currently on this asset. Possible values are:

        :1: Reconnaissence - the attacker must first gather information about
            this target
        :2: Initial Asset - the attacker tries to get an initial foothold onto
            the asset
        :3: Execution - the attacker is already on the asset and tries to do
            some serious damage

    :`parent_asset`: Discrete(1e10) - integer that indicates the ID of the 
        parent asset (e.g. network infrastrucutre for a workstation)
    :`child_assets`: Sequence(Discrete(1e10)) - a tuple of integers that
        indicate the IDs of child assets (e.g. multiple workstations for a 
        LAN device)
    :`exposed`: MultiBinary(3) - 3 boolean values that indicate which damage
        scale of the asset is currently exposed; in the order "C", "I", "A"
    :`damage`: MultiDiscrete(nvec=[4,4,4], start=[0,0,0]) - 3 integer values
        that indicate the current damage of the asset on the 3 damage scales 
        'C', 'I', 'A'. Possible values are from 0-3 for each scale independently
    :`attack_vectors`: Sequence(Discrete(1e10)) - list of inters that resemble 
        asset IDs. This list indicates which other assets are exposed, once the
        current asset is taken over (3 'I' damage). 
    :`dependencies`: Sequence(Discrete(1e10)) - list of integers that resemble
        asset IDs. This list inidcates which other assets automatically inherit
        damage, once the current asset is taken over (3 'I' damage). This field
        is additional information for the agent in comparison to a human player,
        as it is not displayed on the web interface.
    :`active_exploits`: Sequence(Dict(...)) - list of equipment dictionaries of
        all exploits that are currently active on the asset. This is redundant, 
        yet better structured information, as it can also be collected via all 
        equipment of all actions in 'played_actions'.
    :`permanent_effects`: Sequence(Dict(...)) - list of effect dictionaries of
        all permanent effects that are currently active on the asset. This
        is redundant, yet better structured information, as it can also be 
        collected via all effects of all actions in 'played_actions'.
    :`played_actions`: Sequence(Dict(...)) - list of action dictionaries of all
        actions that have been played onto the asset. 
    :`shield`: Discrete(3) - integer that indicates how strong the current asset
        is shielded for future damage

    Action Card Space
    .................

    Actions are represented via a dictionary (space) and mostly used in the 
    `'hand'` field of the general observation space.

    :`card_type`: Discrete(2) - whether the action card is a main action or a
        support action. 

        :0: main action card
        :1: support action card

    :`actor_type`: Discrete(2) - whether the action card is for attacker or 
        defender. Possible values:

        :0: attacker
        :1: defender

    :`target_type`: Discrete(3) - Possible values:

        :0: untargeted - the action does not affect any asset (e.g. it affects
            the oponent role directly)
        :1: multi-targeted - the action affects more than one but also possibly
            no asset
        :2: single-targeted - the action affects exactly one asset
        

    :`def_type`: Discrete(4) - Type of defense action. This value is only set
        for actions with  Possible values:

        :0: not set
        :1: Detection
        :2: Prevention
        :3: Response

    :`template_id`: Discrete(1e10) - ID of the action template. Action 
        templates are used behind the scenes for all information about an 
        action before it is played.
    :`effects`: Sequence(Dict(...)) - A sequence of dictionary spaces, 
        reselmbling effects, elements of the effect space
    :`impact`: MultiDiscrete(nvec=[7, 7, 7], start=[-3, -3, -3]) - The damage
        impact the action has onto an asset. Elements are a 3-Tuple ranging from
        -3 to 3 for each entry, where positive values mean inflicting damage to
        an asset an negative values 'healing'/removing damage from an asset.
    :`soph_requirement`: Discrete(6) - An integer resembling the amount of
        sophistication/skill a role requires before it is able to play the
        action card.
    :`asset_categories`: Sequence(Discrete(20)) - A sequence of integers that 
        indicate on which asset categories (e.g. web server, etc.) the action 
        card can be played onto.
    :`attack_stage`: Discrete(4) - An integer that indicates which attack stage
        an attacker needs to be on an asset, before the action card played onto
        the asset.
    :`oses`: Sequence(Discrete(7)) - A sequence of integers that indicate onto
        which operating systems of assets an action is playable.
    :`success_chance`: Box(0.0, 10.0, dtype=np.float32) - A float value
        indicating the base probability, how likely it is that the action will
        be played successfully.
    :`detection_chance`: Box(0.0, 10.0, dtype=np.float32) - A float value
        indicating the base probability, how likely it is that the action will
        be detected by the oponent (defender) after it was played successfully.
    :`detection_chance_failed`: Box(0.0, 10.0, dtype=np.float32) - A float value
        indicating the base probability, how likely it is that the action will
        be detected by the oponent (defender) after it was played unsuccessfully.
    :`predefined_attack_mask`: Text(0, 3, charset="CIA") - A string indicating
        a predefined attack mask with which the action will be played 
        automatically. If no predefined attack mask is set, the player can 
        choose a damage scale onto which the damage of the action is applied.
        Possible values:

        :"": empty attack mask, no damage scale of the action is applied to an
            asset
        :"C": only the 'confidentiality' damage scale of the action is applied
            to an asset
        :"I": only the 'integrity' damage scale of the action is applied to 
            an asset
        :"A": only the 'availability' damage scale of the action is applied
            to an asset
        :"CI": the 'confidentiality' and the 'integrity' damage scale of the 
            action is applied to an asset
        :"CA": the 'confidentiality' and the 'availability' damage scale of 
            the action is applied to an asset
        :"IA": the 'integrity' and the 'availability' damage scale of the 
            action is applied to an asset
        :"CIA": all three damage scales are applied to an asset


    :`requires_attack_mask`: Discrete(3) - An integer, indicating whether an 
        attack mask can be specified by the player, or is already predefined.
        Possible values:

        :0: not set
        :1: does not require an attack mask from the player, it is pre-defined
        :2: the player can provide an attack mask 

    :`transfer_effects`: Sequence(Dict(...)) - A sequence of dictionary spaces, 
        reselmbling effects, elements of the effect space, that transfer onto a
        main action cards and are activated from their context. This field is 
        usually only set with support action cards.
    :`possible_actions`: Sequence(Discrete(1e10)) - A sequence of integers,
        resembling action template IDs that indicate which support action card
        is compatible with which main action card. This field is usually only
        set for support action cards.


    Equipment Card Space
    ....................
      
    Equipment are represented via a dictionary (space) and mostly in the 
    `'equipment'` field of the general observation space. 

    :`type`: Discrete(13) - integer that represents the type of equipment. 
        Possible values are:

        :1: Attack Tool - a global, permanent, attack equipment. This is a 
            passive equipment, meaning it already provides it's effects 
            automatically when the attacker simply posses it.
        :2: Malware - a local, single-use, attack equipment. This is an active
            equipment, meaning it must be played alongside a main action card.
        :3: Exploit - a local, permanent, attack equipment. This is an active
            equipment, meaning it must be played alongside a main action card.
        :4: Credentials - a local, single-use, attack equipment. This is an 
            active equipment, meaning it must be played alongside a main action 
            card.
        :5: Security System - a global, permanent, defense equipment. This is a 
            passive equipment, meaning it already provides it's effects 
            automatically when the attacker simply posses it.
        :6: Failover - a local, single-use, defense equipment. This is an active
            equipment, meaning it must be played alongside a main action card.
        :7: Fix - a local, single-use, defense equipment. This is an active
            equipment, meaning it must be played alongside a main action card.
        :8: Policy - a global, permanent, defense equipment. This is a 
            passive equipment, meaning it already provides it's effects 
            automatically when the attacker simply posses it.
    
        Integers not listed above are currently unused but reserved for future
        use.
    :`actor_type`: Discrete(3) - integer that indicates which type of actor the 
        equipment belongs to. Possible values are:

        :1: Attacker - equipment can only be used by attackers
        :2: Defender - equipment can only be used by defenders

    :`timing_type`: Discrete(3) - integer that indicates which timing type the 
        equipment belongs to. Possible values are:

        :0: None - the type is not set
        :1: Single Use - equipment can only be used once and is then consumed
        :2: Permanent - equipment can be applied multiple times

    :`scope_type`: Discrete(3) - integer that indicates which scope type the 
        equipment belongs to. Possible values are:

        :0: None - the type is not set
        :1: Local - equipment applies only to (an) action(s) on a single asset
        :2: Global - equipment can be applied multiple times

    :`effects`: Sequence(Dict(...)) - list of effect dictionaries of all effects
        the equipment has.
    :`transfer_effects`: Sequence(Dict(...)) - list of effect dictionaries 
        of all transfer effects that the equipment has. Transfer effects move
        to the main action and are later executed in the context of the main
        action instead of the equipment.
    :`price`: Box(-100.0, 100.0, dtype=np.float32) - price of the equipment.
    :`possible_actions`: Sequence(Discrete(1e10)) - sequence of possible action 
        card IDs that the equipment can be played alongside with.
    :`impact`: MultiDiscrete(nvec=[7, 7, 7], start=[-3, -3, -3]) - impact of 
        the equipment on the three damage scales 'C', 'I', 'A'. Possible values
        range from -3 to 3.
    :`active`: Discrete(3) - indicates whether the equipment is active or not.
        Possible values are:

        :0: not set
        :1: False
        :2: True

        
    Effect Space
    ............

    :type: Discrete(13) - An integer, indicating which type of effect the
        current object is. Each effect has individual properties and influences
        on the gameplay. Possible values:

        :0: Base effect - a dummy type for all effects; has no specific 
            influence on the game whatsoever
        :1: reserverd - this effect type should never be present
        :2: Stop permanent effect - stops a permanent effect of being still 
            active in the next turn
        :3: reserverd - this effect type should never be present
        :4: Inc-dec effect - increases or decreases an attribute of an action
            card or a property of a role (e.g. success chance, credits)
        :5: Permanent inc-dec effect - continously increases or decreases an
            attribute of an action or a property of a role for multiple turns.
            This effect type is a permanent effect.
        :6: Damage shield effect - prevents damage from (oponent) action cards
            to actually inflict damage onto the asset. Has only a certain
            volumina of damage points it can absorb, afterwards the damage is 
            passed through onto a possible other shield or the asset directly.
            Damage absorbtion may be stochastically. This effect type is a 
            permanent effect. 
        :7: Reveal assets effect - reveals an asset on the board that was
            previously hidden to the player (usually attacker)
        :8: Grant equipment effect - grants a free equipment to a player 
        :9: Placeholder effect - placeholder for future effects

    :scope: Discrete(6) - determines the scope onto which targets the effect is
        active. Usually any effect is played alongside an action or is on an 
        equipment that is possesed by a player. Therefore relative scopes are
        always relative to the role that played/posses the effect. In a 1v1 game
        some of these scopes are redundant. However in a MvN player game these
        scopes make a difference and this implementation already prepares MvN
        player games.
        Possible values:

        :1: attackers - effect applies to all targets that are possesd/played by
            attackers.
        :2: defenders - effect applies to all targets that are possesd/played by
            defenders.
        :3: own - effect applies to all targets that are possesed/played by the
            same player that activates this effect (e.g. success chance bonus
            for future own actions played)
        :4: not own - effect applies to all targets that are possesed/played by
            other roles (e.g. success chance penalty for future actions of 
            opponents)
        :5: all - effect applies to all targets

    :owner_id: Discrete(1e10) - ID of the role in the game that owns the effect.
        This ID is used to apply effects based on the relative scopes 'own' and
        'not own'. 
    :attributes: Sequence(Text(max_len=300)) - A sequence of attribute names
        that are changed by inc-dec effects. This field is only set for inc-dec
        and permanent inc-dec effects.
    :value: Box(-100.0, 100.0, dtype=np.float32) - a float value indicating 
        attribute value shifts for inc-dec values or amount of damage absorbtion
        for shield effects.
    :active: Discrete(3) - integer indicating whether the effect is currently
        active or not. This field is only relevant for permanent effects.
    :turns: Discrete(MAX_AMOUNT_OF_TURNS+1, start=-1) - an integer indicating
        the amount of turns the effect is active. -1 indicates the effect being
        active for infinietly many turns. This field is only relevant for
        permanent effects.
    :num_effects: Discrete(1000) - An integer, indicating how many effects can
        be stopped by the current effect. This field is only relevant for stop
        permanent effects. 
    :probability: Box(0.0, 5.0, dtype=np.float32) - a float value, indicating
        the likelyhood of an active being activated for stochastic effects.
        This field is only relevant for damage shield effects. 
    :equipment: Sequence(effect_equipment_space) - a sequence of equipment the
        effect grants the player. This field is only relevant with grant
        equipment effects. 
    :damage: MultiDiscrete(nvec=[7, 7, 7], start=[-3, -3, -3]) - Amount of
        damage the effect applies. This field is currently not relevant and
        reserved for future use.

        
    Actor Space
    ...........

    Some of the fields in the actor space (especially the fields with text 
    spaces) are just for very advanced agents that might extract information
    from e.g. the mission description or the role description. 

    .. note::

        The terms 'actor' and 'role' are used interchangeably.

    :connection_id: Text(max_len=500) - a UUID of the connection of the player 
        that is playing the corresponding role
    :id: Discrete(64) - an integer that identifies of the role within the game.
    :type: Discrete(3) - an integer that defines the type of the actor. Possible
        values:

        :0: not set
        :1: attacker
        :2: defender

    :soph: Discrete(10) - an integer that indicates the level of sophistication
        or skill (these two terms are also used interchagably) the role has. 
        Some actions require a certain level of skill in order to be played.
        Therefore the level of this attribute has an influence on how many
        different actions the role can play. Also if a role has significantly 
        more skill than the required level, the probability of success gets 
        increased.
    :det: Discrete(10) - an integer that determines the level of determination
        of the role. The value of this attribute has an influence on how many 
        action cards the role can hold on its hand at the same time.
        Additionally it also influences initiative value of the role. 
    :wealth: Discrete(10) - an integer that influences the initial amount of
        credits the role has, as well as for defenders the amount of credits the
        role receives each turn.
    :ins: Discrete(30) - an integer that indicates the amount of knowledge the
        actor has about its opponent. The higher this value is, the more likely
        it is that own actions are successful and opponent actions are detected.
    :ini: Discrete(16) - an integer indicating the initiave of an attacker. 
        Each main action that is played costs 1 initiative point as soon as the
        defender has detected the first action of the attacker. If the 
        initiative value drops to 0, the game ends and the attacker looses the
        game.
    :credits: Box(low=0.0, high=30.0, dtype=np.float32) - a float value
        indicating how many credtis (the in-game currency) the role currently
        holds. Credits can be used to purchase equipment that helps the player
        during the game. Some actions may also require credits while some
        equipment may bring an attacker credits.
    :mission_description: Text(max_len=5000) - a short description about the
        mission of the player.
    :goal_descriptions: Sequence(Text(max_len=2000) - a seuqences of strings
        that individually describe the goals of the attacker within the 
        scenario. This field is only relevant for attackers.
    :goals: Sequence(Dict(...)) - A sequence of dicitonaroies of goals an
        attacker needs to achieve in order to win the game of the following
        format: 
        
        :type: Discrete(4) - an integer describing the type of the goal the
            attacker has to achive. Possible values:

            :1: asset goal - a specific asset needs to have certain attribute
                values in order for the attacker to achieve this goal (e.g. 
                damage)
            :2: actor goal - attribute values of the attacker itself need to
                reach certain values in order to achieve this goal (e.g. 
                insight).
            :3: defender not exceeded actor goal - the defender must not reach
                a specific value in order for the attacker to achieve this goal

        :asset_id: Discrete(1e10) - an integer indicating the ID of the asset
            within the scenario that is affected by the goal. This field is only
            relevant for asset goals.
        :damage: MultiDiscrete(nvec=[4, 4, 4]) - The damage the asset is
            required to have at least in order for the attacker to achieve the
            goal. This field is only relevant for asset goals.
        :exposed: MultiBinary(3) - The exposedness of the asset that is
            required to have at least in order for the attacker to achieve the
            goal. This field is only relevant for asset goals.
        :attack_stage: Discrete(4) - The exposedness of the asset that is
            required to have at least in order for the attacker to achieve the
            goal. This field is only relevant for asset goals.
        :credits: Box(low=0.0, high=30.0, dtype=np.float32) - a float value 
            indicating the amount of credits an attacker needs to have or a 
            defender should not exceed to have in order for an attacker to 
            achieve the goal. This field is only relevant for actor goals or 
            defender not exceeded actor goals.
        :ins: Discrete(10) - an integer indicating the amount of insight an 
            attacker needs to have or a defender should not exceed to
            have in order for an attacker to achieve the goal. This field is 
            only relevant for actor goals or defender not exceeded actor goals.
        :defender: Discrete(64) - an integer indicating the ID of the role
            within the scenario that is affected by a 

        This field is only relevant for attackers, as defenders always have the
        same goal, namely to stop the attacker from achieveing their goals

    :assets: Sequence(Discrete(1e10)) - a sequence of integers that indicate
        which assets on the board belong to the defender. This field is 
        only relevant for defenders.

    
    Rewards
    -------
    
    A reward of '+1' is given if the game was won by the agent at the terminal
    state once a game has ended. If the agent lost, a '-1' is given to the
    agent. On all intermediate stepts a reward of '0' is given. 

    
    Starting State  
    --------------
    The starting state heavily depends on the selected scenario the agent plays.
    However under default game options, the game will start in a shopping phase
    leaving the agent to buy equipment. 

    
    Episode End
    -----------
    An episode ends, either when the attacker has fulfilled all of its goals,
    or if the attacker's initiative reached 0.

    
    Arguments
    ---------
    PenQuest Environment currently acceptes 3 arguemnts: `options`, 
    `render_mode`, and `config_file_path`. Note that currently no render modes
    are accepted and that env.render() raises a `NotImplemented` error. 

    For more information on `options` please refer to the GameOptions 
    documentation.

    `config_file_path` accepts a file path to a config file that stores
    configuration information. For more information on this please refer to the
    configuraiton files documentation.


    Additional Information
    ----------------------
    Each step, the PenQuest environment provides additionl information to the
    agent. These are:

    :interaction_type: the current type of interaction that is required for the
        agent. Possible values are:

        :0-2: reserved for game initialisation
        :3: shopping phase - the agent needs to buy equipment or skip the
            shopping phase
        :4: play card - the agent needs to play an action card from its hand
        :5: choose action - the agent needs to re-draw an action card from a
            selection
        :6: end - the game has ended
    
    :valid_actions: a list of tuples that indicate all currently playable
        actions. For the format of valid actions, please refer to the action
        space section.
    :end_state: an integer value that indicates how the game ended (won, lost,
        draw, surrender).


    """
    metadata: Dict[str, Any] = {"render_modes":[]}

    def __init__(
            self, 
            options: Dict, 
            render_mode: Optional[str]=None, 
            config_file_path: str=None
        ):
        """Initializes all attributes

        :param options: game options that determine different types of PenQuest 
            games
        :param render_mode: the way the environment is rendered. This is
            currently not supported, defaults to None
        :param config_file_path: file path to a config file, defaults to 
            'default_config.ini'
        :raises RuntimeError: Invalid game Option
        """
        self.last_interaction_type: GameInteractionType = None
        self.step_num: int = 0
        self.valid_actions: Set[Tuple] = {}

        if config_file_path is None:
            full_path = os.path.dirname(os.path.abspath(__file__))
            full_path = full_path.replace("/penquest_env/penquest_env", "/penquest_env")
            config_file_path = os.path.join(full_path, DEFAULT_CONFIG_FILE)

        self.config_file_path = config_file_path
        self.options = options if options is not None else dict()
        if self.options.get(KEY_INITIAL_ACTION_MODE, 0) == 1:
            raise RuntimeError(
                "Initial Action Mode 1 is currently not supported in the "
                "environment"
            )
        if self.options.get(KEY_EQUIPMENT_SHOP_MODE, 1) == 2:
            raise RuntimeError(
                "Equipment Shop Mode 2 is currently not supported in the "
                "environment"
            )

        # gymnasium attributes
        self.observation_space = self._get_obs_space()
        self.action_space = Sequence(Discrete(int(1e10)))
        self.reward_range = (-1.0, 1.0)
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # These need to be set in reset() because they depend on the game
        self.game: Game = None
        self.connetor: ConnectionHelper = None
        self.obs_factory: ObservationFactory = None

    def _get_obs_space(self):
        """define observation space"""
        self.actor_space = gym.spaces.Dict(
            {
                "type": Discrete(3),
                "id": Discrete(MAX_AMOUNT_OF_ACTORS_IN_GAME),
                "connection_id": Discrete(10),
                "soph": Discrete(10),
                "det": Discrete(10),
                "wealth": Discrete(10),
                "ins": Discrete(30),
                "ini": Discrete(16),
                "credits": Box(low=0.0, high=30.0, dtype=np.float32),
                # future agents might infer information about the 
                # mission description or the goal descriptions
                "mission_description": Text(
                    min_length=0, 
                    max_length=5000,
                    charset=CHARSET
                ),
                "goal_descriptions": Sequence(
                    Text(min_length=0, max_length=2000, charset=CHARSET)
                ),
                # Only relevant for attackers
                "goals": Sequence(
                    gym.spaces.Dict({
                        "type": Discrete(MAX_AMOUNT_OF_GOAL_TYPES),
                        "asset_id": Discrete(MAX_AMOUNT_OF_ASSETS),
                        "damage": MultiDiscrete(nvec=[4, 4, 4]),
                        "exposed": MultiBinary(3),
                        "attack_stage": Discrete(4),
                        "credits": Box(low=0.0, high=30.0, dtype=np.float32),
                        "ins": Discrete(10),
                        "defender": Discrete(MAX_AMOUNT_OF_ACTORS_IN_GAME),
                    })
                ), 
                # Only relevant for defenders
                "assets": Sequence(Discrete(MAX_AMOUNT_OF_ASSETS))
            }
        )
        # These two additional equipment/effect spaces are necessary, because
        # otherwise there would be a referencing circle in the obs space.
        # Note that this setup has problems with scenarios where actions have
        # grant-equipment effects with equipment that itself has a grant-
        # equipment effect.
        no_equipment_effect_space = gym.spaces.Dict({
            "type": Discrete(AMOUNT_OF_EFFECT_TYPES),
            "scope": Discrete(6),
            "owner_id": Discrete(MAX_AMOUNT_OF_ACTORS_IN_GAME),
            "attributes": Sequence(Text(max_length=300, charset=CHARSET)),
            "value": Box(-100.0, 100.0, dtype=np.float32),
            "active": Discrete(3),
            "turns": Discrete(MAX_AMOUNT_OF_TURNS+1, start=-1),
            "num_effects": Discrete(1000),
            "probability": Box(0.0, 5.0, dtype=np.float32),
            ##"equipment": Sequence(equipment_space),
            "damage": MultiDiscrete(nvec=[7, 7, 7], start=[-3, -3, -3]),
        })
        effect_equipment_space = gym.spaces.Dict({
            "type": Discrete(13),
            "actor_type": Discrete(3),
            "timing_type": Discrete(3), # permanent equipment or single use
            "scope_type": Discrete(3), # global equipment or local
            "effects": Sequence(no_equipment_effect_space),
            "transfer_effects": Sequence(no_equipment_effect_space),
            "price": Box(-100.0, 100.0, dtype=np.float32),
            "possible_actions": Sequence(Discrete(MAX_AMOUNT_OF_ACTION_TEMPLATES)),
            "impact": MultiDiscrete(nvec=[7, 7, 7], start=[-3, -3, -3]),
            #"used_on": Discrete(MAX_AMOUNT_OF_ASSETS),
            "active": Discrete(3)
        })
        self.effect_space = gym.spaces.Dict({
            "type": Discrete(AMOUNT_OF_EFFECT_TYPES),
            "scope": Discrete(6),
            "owner_id": Discrete(MAX_AMOUNT_OF_ACTORS_IN_GAME),
            "attributes": Sequence(Text(max_length=300, charset=CHARSET)),
            "value": Box(-100.0, 100.0, dtype=np.float32),
            "active": Discrete(3),
            "turns": Discrete(MAX_AMOUNT_OF_TURNS+1, start=-1),
            "num_effects": Discrete(1000),
            "probability": Box(0.0, 5.0, dtype=np.float32),
            "equipment": Sequence(effect_equipment_space),
            "damage": MultiDiscrete(nvec=[7, 7, 7], start=[-3, -3, -3]),
        })
        self.equipment_space = gym.spaces.Dict({
            "type": Discrete(13),
            "actor_type": Discrete(3),
            "timing_type": Discrete(3), # permanent equipment or single use
            "scope_type": Discrete(3), # global equipment or local
            "effects": Sequence(self.effect_space),
            "transfer_effects": Sequence(self.effect_space),
            "price": Box(-100.0, 100.0, dtype=np.float32),
            "possible_actions": Sequence(Discrete(MAX_AMOUNT_OF_ACTION_TEMPLATES)),
            "impact": MultiDiscrete(nvec=[7, 7, 7], start=[-3, -3, -3]),
            #"used_on": Discrete(MAX_AMOUNT_OF_ASSETS),
            "active": Discrete(3)
        })
        self.actions_space = gym.spaces.Dict({
            #"type": Discrete(5),
            "card_type": Discrete(2),
            "actor_type": Discrete(2),
            "target_type": Discrete(3),
            "def_type": Discrete(4),
            "template_id": Discrete(MAX_AMOUNT_OF_ACTION_TEMPLATES),
            "effects": Sequence(self.effect_space),
            "impact": MultiDiscrete(nvec=[7, 7, 7], start=[-3, -3, -3]),
            "soph_requirement": Discrete(6),
            "asset_categories": Sequence(Discrete(MAX_AMOUNT_OF_ASSET_CATEGORIES)),
            "attack_stage": Discrete(4),
            "oses": Sequence(Discrete(MAX_AMOUNT_OF_OSES)),
            "success_chance": Box(0.0, 10.0, dtype=np.float32),
            "detection_chance": Box(0.0, 10.0, dtype=np.float32),
            "detection_chance_failed": Box(0.0, 10.0, dtype=np.float32),
            "predefined_attack_mask": Text(
                min_length=0, 
                max_length=3, 
                charset="CIA"
            ),
            "requires_attack_mask": gym.spaces.Discrete(3),
            "transfer_effects": Sequence(self.effect_space),
            "possible_actions": Sequence(Discrete(MAX_AMOUNT_OF_ACTION_TEMPLATES))
        })
        
        self.asset_space = gym.spaces.Dict({
            "id": Discrete(MAX_AMOUNT_OF_ASSETS),
            "category": Discrete(MAX_AMOUNT_OF_ASSET_CATEGORIES),
            "os": Discrete(MAX_AMOUNT_OF_OSES),
            "attack_stage": Discrete(MAX_AMOUNT_OF_ATTACK_STAGES),
            "parent_asset": Discrete(MAX_AMOUNT_OF_ASSETS),
            "child_assets": Sequence(Discrete(MAX_AMOUNT_OF_ASSETS)),
            "exposed": MultiBinary(3),
            "damage": MultiDiscrete(nvec=[4, 4, 4], start=[0, 0, 0]),
            "attack_vectors": Sequence(Discrete(MAX_AMOUNT_OF_ASSETS)),
            "dependencies": Sequence(Discrete(MAX_AMOUNT_OF_ASSETS)),
            "active_exploits": Sequence(self.equipment_space),
            "permanent_effects": Sequence(self.effect_space),
            "played_actions": Sequence(self.actions_space),
            "shield": Discrete(3)
        })
        observation_space = gym.spaces.Dict(
            {
                "turn": Discrete(MAX_AMOUNT_OF_TURNS),
                "phase": Discrete(6),
                "actor_id": Discrete(MAX_AMOUNT_OF_ACTORS_IN_GAME),
                "actor_connection_id": Discrete(MAX_AMOUNT_OF_ACTORS_IN_GAME),
                "roles": Sequence(self.actor_space),
                "hand": Sequence(self.actions_space),
                "equipment": Sequence(self.equipment_space),
                "board": Sequence(self.asset_space),
                "shop": Sequence(self.equipment_space),
                "selection_choices": Sequence(self.actions_space),
                "selection_amount": Discrete(MAX_AMOUNT_SELECTION)
            }
        )
        return observation_space
    
    def _get_obs(self) -> Dict:
        """Creates an observation from the current game state and checks whether
        it applies to the speciefied observation space. 

        Returns:
            Dict: the observation according to the observation space
        """
        observation = self.obs_factory.create_observation(self.game.game_state)
        for role in observation["roles"]:
            assert role in self.actor_space
        for action in observation["hand"]:
            for effect in action["effects"]:
                assert effect in self.effect_space
            assert action in self.actions_space
        for action in observation["selection_choices"]:
            for effect in action["effects"]:
                assert effect in self.effect_space
            assert action in self.actions_space
        for equipment in observation["equipment"]:
            for effect in equipment["effects"]:
                assert effect in self.effect_space
            assert equipment in self.equipment_space
        for equipment in observation["shop"]:
            for effect in equipment["effects"]:
                assert effect in self.effect_space
            assert equipment in self.equipment_space
        for asset in observation["board"]:
            for equipment in asset["active_exploits"]:
                assert effect in self.effect_space
            for effect in asset["permanent_effects"]:
                assert effect in self.effect_space
            for action in asset["played_actions"]:
                assert action in self.actions_space
            assert asset in self.asset_space
        assert observation in self.observation_space
        return observation

    async def _get_valid_actions(self) -> Set[Tuple[int]]:
        valid_actions = []
        if self.last_interaction_type == GameInteractionType.SHOPPING_PHASE:
            role = self.game.get_player_role()
            def add_affordable_item(all_items: List, limit:float, current_selection):
                valid_selections = []
                current_sum = sum([item.price for item in current_selection])
                if all([item in current_selection or current_sum + item.price > limit for item in all_items]):
                    return [tuple([all_items.index(item) for item in current_selection])]
                for item in all_items:
                    if item not in current_selection and current_sum + item.price <= limit:
                        valid_selections += add_affordable_item(all_items, limit, current_selection+[item])
                return valid_selections

            valid_actions = tuple(
                add_affordable_item(self.game.game_state.shop, role.credits, [])
            )
        elif self.last_interaction_type == GameInteractionType.CHOOSE_ACTION:
            selection_len = len(self.game.game_state.selection_choices)
            selection_amount = self.game.game_state.selection_amount
            valid_actions = list(permutations(range(selection_len), selection_amount))
        elif self.last_interaction_type == GameInteractionType.PLAY_CARD:
            valid_actions = await self.game.get_valid_actions()
        else:
            get_logger(__name__).error(
                f"Unknown interaction type {self.last_interaction_type}"
            )
        self.valid_actions = valid_actions
        return valid_actions

    async def _get_info(self, get_valid_actions: bool = True) -> Dict[str, Any]:
        """Creates the additional information dictionary with the fields:
        - interaction_type: the type of action the agent needs to pass; usually
            3 - for shopping, 4 - for playing an action from the hand, 5 - 
            select an action to draw to the hand
        - valid_actions: a set of tuples that list all valid actions in the 
            current state
        - victory: boolean value that indicates whether the agent won the game

        Returns:
            Dict[str, Any]: _description_
        """
        if get_valid_actions:
            valid_actions = await self._get_valid_actions()
        else:
            valid_actions = {}
        victory = False
        if self.game is not None:
            if self.game.game_state is not None:
                if self.game.game_state.end_state is not None:
                    victory = self.game.game_state.end_state == GameEndedState.WON
        info = {
                "interaction_type": self.last_interaction_type.value if self.last_interaction_type is not None else 0,
                "valid_actions": valid_actions,
                "victory": victory
        }
        return info

    def _get_reward(self) -> float:
        if not self.game.is_over():
            return 0.0
        return 1.0 if self.game.game_state.end_state == GameEndedState.WON else -1.0

    def reset(self, seed:int=None, options:Dict={}):
        """_summary_

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        super().reset(seed=seed)
        rand = np.random.RandomState(seed)
        if options is None or len(options) == 0:
            options = self.options
        self.step_num = 0
        self.last_interaction_type = None

        # reset the game (this function will be run blocking)
        async def reset_game():
            """Establishes a new game ready to play

            :raises ValueError: Inconsistent State Error - the agent should be in
                the lobby but isn't found
            :return: tuple of first observation and info object; is None, None
                if an error occurd/the game was closed or anything else happend
                that where the game could not be started
            """
            self.game = Game()
            self.connector = ConnectionHelper(self.game)
            self.obs_factory = ObservationFactory()
            await self.connector.connect_to_server(self.config_file_path)

            done = False
            while not done:
                # get interaction type
                interaction_type = await self.game.next_interaction_type()
                if interaction_type is None:
                    done = True
                    continue

                # switch n case and handle different interaction types
                if interaction_type == GameInteractionType.CREATE_OR_JOIN_LOBBY:
                    get_logger(__name__).debug("Create or join lobby interaction")

                    # check if lobby should be joined
                    game_code = options.get('join', None)
                    if game_code is not None:
                        await self.game.join_game(game_code)
                    else: # else create lobby
                        if 'scenarios' in options:
                            scenarios = options['scenarios']
                            scenario = rand.choice(scenarios)
                        else:
                            scenario = options.get('scenario', DEFAULT_SCENARIO)
                        try:
                            await self.game.create_new_lobby(
                                scenario, 
                                options.get('game_options', dict())
                            )
                        except asyncio.TimeoutError as e:
                            get_logger(__name__).error(
                                f"Error while waiting for slot to be "
                                f"changed in step {self.step_num}: {e}"
                            )
                            await self.game.close()
                            continue
                elif interaction_type == GameInteractionType.CHANGE_LOBBY_PROPERTIES:
                    get_logger(__name__).debug(
                        "Lobby properties interaction like adding players / "
                        "bots or chaning roles!"
                    )

                    # change slot of agent
                    slot = options.get('slot', None)
                    if slot is not None:
                        # TODO: Change to connection ID
                        current_own_slots = [
                            j 
                            for j, player in self.game.lobby.players.items() 
                            if player.connection_id == self.game.actor_connection_id
                        ]
                        if len(current_own_slots) <= 0: 
                            raise ValueError(
                                "InconsistenStateError: Agent is currently not "
                                "in the lobby, although it should"
                            )
                        current_own_slot = current_own_slots[0]
                        if slot != current_own_slot:
                            try:
                                await self.game.change_slot(slot)
                            except asyncio.TimeoutError as e:
                                get_logger(__name__).error(
                                    f"Error while waiting for slot to be "
                                    f"changed in step {self.step_num} in game "
                                    f"{self.game.lobby.code}: {e}"
                                )
                                await self.game.close()
                                continue

                    # add bots and wait for players
                    players_config = options.get('players', None)
                    player_timeout = options.get(
                        'wait_for_players', 
                        DEFAULT_WAIT_FOR_PLAYERS_PERIOD
                    )
                    if players_config is not None:
                        # add as many bots as in the config
                        for bot in [it for it in players_config if it['type'] == 'bot']:
                            await self.game.add_bot(bot_type=bot['bot_type'])
                        
                        try:
                            await self.game.wait_for_players(
                                len(players_config), 
                                player_timeout
                            )
                        except asyncio.TimeoutError as e:
                            get_logger(__name__).error(
                                f"Error while waiting for player in step "
                                f"{self.step_num} in game "
                                f"{self.game.lobby.code}: {e}"
                            )
                            await self.game.close()
                    
                    # set seed
                    seed = options.get('seed', None)
                    if seed is not None:
                        await self.game.set_seed(seed)

                    goal = options.get('goal', None)
                    if goal is not None:
                        await self.game.set_goal(goal)

                elif interaction_type == GameInteractionType.PLAYER_READY:
                    get_logger(__name__).debug("Player ready interaction!")

                    # set player ready to start the game
                    await self.game.set_player_readiness()
                    done = True
                    continue
                elif interaction_type == GameInteractionType.END:
                    get_logger(__name__).debug("Interactions ended!")
                    break
                else:
                    get_logger(__name__).error(
                        f"Unknown interaction type {self.last_interaction_type}"
                    )
            if interaction_type != GameInteractionType.END:
                interaction_type = await self.game.next_interaction_type()
                self.last_interaction_type = interaction_type
                observation = self._get_obs()
                try:
                    info = await self._get_info()
                except asyncio.TimeoutError as e:
                    get_logger(__name__).error(
                        f"Error while waiting for valid actions in step "
                        f"{self.step_num} in game {self.game.game_state.name}: "
                        f"{e}"
                    )
                    await self.game.close()
                    return None, None
                return observation, info
            else:
                raise RuntimeError("Could not establish a running game")
            
        
        loop = asyncio.get_event_loop()
        obs, info = loop.run_until_complete(reset_game())
        return obs, info
    
    async def _shop_equipment(self, action: Tuple[int]):
        """Buys the equipment that was provided in the action

        :param action: tuple of integers that specify which euqipment should be 
            bought; integers specifiy indices of equipment in the current 'shop'
            list
        """
        equipment_ids = [self.game.game_state.shop[idx].id for idx in action]
        if len(equipment_ids) > 0:
            # Buy equipment command already has a flag that ends the shopping
            # phase
            await self.game.buy_equipment(equipment_ids)
        else:
            await self.game.finish_shopping()

    async def _choose_action(self, action: Tuple[int]):
        """Choses actions to draw from a pre-received offer of multiple possible
        actions

        :param action: tuple of integers that specify which actions shoudl be
            drawn; integers specify indices of actions in the 
            'selection_choices' list
        """
        action_ids = [
            self.game.game_state.selection_choices[idx].id for idx in action
        ]
        await self.game.selection_choose(action_ids)

    async def _play_card(self, action: Tuple[int]):
        """Plays an action onto the specified target with the according attack
        mask supporting the specified support action and equipment

        :param action: tuple of integers that specify which 
            action/support/equipment/target should be played; integers specify 
            indices into the according lists (e.g. 'hand')
        """
        main_action = self.game.game_state.hand[action[0]]
        target_asset_id = action[1] if action[1] > 0 else None
        attack_mask = ATTACK_MASKS[action[2]]
        if attack_mask == "":
            attack_mask = main_action.predefined_attack_mask
        support_action_ids = [self.game.game_state.hand[action[3]-1].id] if action[3] > 0 else None
        equipment_ids = [self.game.game_state.equipment[action[4]-1].id] if action[4] > 0 else None
        response_target_id = action[5]
        await self.game.play_action(
            main_action.id,
            target_asset_id,
            attack_mask=attack_mask,
            support_action_ids=support_action_ids,
            equipment_ids=equipment_ids,
            response_target_id=response_target_id
        )
        

    def step(self, action: Tuple[int]):

        async def step(action: Tuple[int]):
            terminated = False
            truncated = False

            assert action in self.action_space
            if action not in self.valid_actions:
                # this is an exception instead of an auto-surrender in order 
                # that agent developers can catch it and handle the case
                raise ValueError(
                    f"Invalid Action: action {action} is invalid. Please "
                    "select a valid action"
                )

            try:
                # depending on the interaction type, process the action of the agent
                if self.last_interaction_type == GameInteractionType.SHOPPING_PHASE:
                    await self._shop_equipment(action)
                elif self.last_interaction_type == GameInteractionType.CHOOSE_ACTION:
                    await self._choose_action(action)
                elif self.last_interaction_type == GameInteractionType.PLAY_CARD:
                    await self._play_card(action)
                elif self.last_interaction_type == GameInteractionType.END:
                    get_logger(__name__).debug(f"Interactions ended")
                    truncated = True
                else:
                    get_logger(__name__).error(
                        f"Unknown interaction type {self.last_interaction_type}"
                    )
            except asyncio.TimeoutError as e:
                get_logger(__name__).error(
                    f"Error while waiting for action to be processed in step "
                    f"{self.step_num} in game {self.game.game_state.name}: "
                    f"{e}"
                )
                await self.game.close()
                truncated = True

            # the order of the following steps is sensitive to the correct 
            # working of the environment. First wait for next interaciton type
            # because the "game_ended" message arrives within this phase.
            # The reward builds on the 'victory' field that comes with the
            # 'game_ended' message. Afterwards a new observation needs to be made
            # as well as a new info object, before the game can be left.
            # Finally the game can be left.

            # generate next observation
            if self.last_interaction_type != GameInteractionType.END:
                try:
                    interaction_type = await self.game.next_interaction_type()
                    self.last_interaction_type = interaction_type
                except asyncio.TimeoutError as e:
                    get_logger(__name__).error(
                        f"Error while waiting for shop in step {self.step_num}:"
                        f" {e}"
                    )
                    await self.game.close()
                    truncated = True

            # get reward for previous action
            reward = self._get_reward()

            self.step_num += 1
            new_observation = self._get_obs()

             
            try:
                info = await self._get_info(
                    get_valid_actions=self.last_interaction_type != GameInteractionType.END
                )
            except asyncio.TimeoutError as e:
                get_logger(__name__).error(
                    f"Error while waiting for valid actions in step "
                    f"{self.step_num} in game {self.game.game_state.name}: "
                    f"{e}"
                )
                await self.game.close()
                info = dict()
                truncated = True
            
            # get termination status
            if self.game.is_over():
                terminated = True
                # this is the official, graceful exit!
                await self.game.leave_game()
                await self.game.close()
                self.game = None

            return new_observation, reward, terminated, truncated, info

        
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(step(action))
    
    def render(self):
        """Renders the agents observation of the environment"""
        raise NotImplementedError(
            "There is currently no rendering implemented. However you can play "
            "against the agent from the web interface."
        )
            
    def close(self):
        """Closes all open connections required to use the enviornment"""

        async def close():
            if self.game is not None:
                if self.game.lobby is not None or self.game.game_state is not None:
                    await self.game.leave_game()
                await self.game.close()

        loop = asyncio.get_event_loop()
        return loop.run_until_complete(close())