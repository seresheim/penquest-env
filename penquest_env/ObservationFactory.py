from typing import Dict, Union
from enum import Enum
from bidict import bidict

from penquest_pkgs.model import (
    GameState,
    Action,
    Actor,
    Effect,
    Asset,
    Equipment,
    ActionTemplate,
    EquipmentTemplate
)

import numpy as np



MAP_ACTOR_TYPE = {
    "base_actor": 0,
    "attacker": 1,
    "defender": 2
}

MAP_ACTION_TYPES = {
    "base_action": 0,
    "main_attack_action": 1,
    "support_attack_action": 2,
    "main_defense_action": 3,
    "support_defense_action": 4
}

MAP_ACTION_CARD_TYPES = {
    "main": 0,
    "support": 1
}

MAP_ACTION_ACTOR_TYPES = {
    "attack": 0,
    "defense": 1
}

MAP_ACTION_TARGET_TYPES = {
    "untargeted": 0,
    "multi": 1,
    "single": 2
}

MAP_EFFECT_TYPES = {
    "base_effect": 0,
    "permanent_effect": 1,
    "stop_permanent_effect": 2,
    "base_inc_dec_effect": 3,
    "inc_dec_effect": 4,
    "permanent_inc_dec_effect": 5,
    "damage_shield_effect": 6,
    "reveal_assets_effect": 7,
    "grant_equipment_effect": 8,
    "placeholder_effect": 9
}

PERMANENT_EFFECTS = {1, 5, 6}
INC_DEC_EFFECTS = {3, 4, 5}

MAP_EQUIPMENT_TYPES = {
    "base_equipment": 0,
    "Single-UseEquipment": 1,
    "PermanentEquipment": 2,
    "AttackEquipment":3,
    "DefenseEquipment":4,
    "AttackTool":5,
    "Malware":6,
    "Credentials":7,
    "Exploit": 8,
    "SecuritySystem": 9,
    "Failover": 10,
    "Fix": 11,
    "Policy": 12
}

MAP_SCOPE_TYPES = {
    "Scope.Attackers": 1,
    "Scope.Defenders": 2,
    "Scope.Own": 3,
    "Scope.NotOwn": 4,
    "Scope.All": 5
}

MAP_GOAL_TYPES = {
    "asset_goal": 1,
    "actor_goal": 2,
    "defender_not_exceeded_actor_goal": 3
}

MAP_ATTACK_STAGES = {
    'Reconnaissance': 1,
    'Initial Access': 2,
    'Execution': 3
}

PERMANENT_EQUIPMENT = {2, 5, 8, 9, 12}
SINGLE_USE_EQUIPMENT = {1, 6, 7, 10, 11}
ATTACK_EQUIPMENT = {3, 6, 7, 8}
DEFENSE_EQUIPMENT = {4, 9, 10, 11, 12}
LOCAL_EQUIPMENT = {6, 7, 8, 10, 11, 12}
GLOBAL_EQUIPMENT = {5, 9}

class ActorTypes(Enum):
    Null = 0
    Attack = 1
    Defense = 2

class ScopeTypes(Enum):
    Null = 0
    Local = 1
    Global = 2

class TimingTypes(Enum):
    Null = 0
    Single_Use = 1
    Permanent = 2

class ObservationFactory():

    def __init__(self):
        # IDs only need to be relevant within a game, therefore translating them
        # to an integer that may change between games is sufficient. Also the
        # agent should not learn IDs by hard anyway.
        self.connection_id_mapping = bidict()
        self.actor_id_mapping = bidict()
        self.action_template_id_mapping = bidict()
        self.asset_id_mapping = bidict()
    
    def _get_id(self, actor_id: str, mapping: bidict) -> int:
        if actor_id not in mapping:
            mapping[actor_id] = len(mapping) + 1
        return mapping[actor_id]
    
    def _get_actor_id(self, actor_id: str) -> int:
        return self._get_id(actor_id, self.actor_id_mapping)
    
    def _get_conn_id(self, conn_id: str) -> int:
        return self._get_id(conn_id, self.connection_id_mapping)
    
    def _get_action_template_id(self, action_id: str) -> int:
        return self._get_id(action_id, self.action_template_id_mapping)
    
    def _get_asset_id(self, asset_id: str) -> int:
        return self._get_id(asset_id, self.asset_id_mapping)
        
    
    def create_observation(self, game_state: GameState) -> Dict:
        role_obs = tuple([
            self._create_actor_obs(actor, conn_key) 
            for conn_key, actor in game_state.roles.items()
        ])
        action_obs = tuple([
            self._create_action_obs(action)
            for action in game_state.hand
        ])
        selection_obs = tuple([
            self._create_action_obs(action)
            for action in game_state.selection_choices
        ])
        equipment_obs = tuple([
            self._create_equipment_obs(eq) for eq in game_state.equipment
        ])
        shop_obs = tuple([
            self._create_equipment_obs(eq) for eq in game_state.shop
        ])
        asset_obs = tuple([
            self._create_asset_obs(asset) for asset in game_state.assets_on_board
        ])

        actor_id = 0
        if game_state.actor_id is not None:
            actor_id = self._get_actor_id(game_state.actor_id)

        return  {
                "turn": game_state.turn,
                "phase": game_state.external_phase.value,
                "actor_connection_id": self._get_conn_id(game_state.actor_connection_id),
                "actor_id": actor_id,
                "roles": role_obs,
                "hand": action_obs,
                "equipment": equipment_obs,
                "board": asset_obs,
                "shop": shop_obs,
                "selection_choices": selection_obs,
                "selection_amount": game_state.selection_amount
            }
    
    def _create_actor_obs(self, actor: Actor, conn_str:str):
        actor_obs = {
            "connection_id": self._get_conn_id(conn_str),
            "id": self._get_actor_id(actor.id),
            "type": MAP_ACTOR_TYPE.get(actor.type, 0),
            "soph": int(actor.soph) if actor.soph is not None else 0,
            "det": int(actor.det) if actor.soph is not None else 0,
            "wealth": int(actor.wealth) if actor.soph is not None else 0,
            "ini": int(actor.ini) if actor.soph is not None else 0,
            "ins": int(actor.ins) if actor.soph is not None else 0,
            "credits": np.array([actor.credits], dtype=np.float32) if actor.soph is not None else np.array([0.0], dtype=np.float32),
            "mission_description": actor.mission_description if actor.mission_description is not None else "",
            "goal_descriptions": tuple(actor.goal_descriptions) if actor.soph is not None else tuple(),
            "goals": tuple(),
            "assets": tuple()
        }
        goals = []
        if actor.goals is not None:
            for goal in actor.goals:
                if goal.type == "asset_goal":
                    goal_obs = {
                        "type": MAP_GOAL_TYPES.get(goal.type, 0),
                        "asset_id": goal.asset.id,
                        "damage": np.array(goal.damage),
                        "exposed": goal.exposed,
                        "attack_stage": MAP_ATTACK_STAGES[goal.attack_stage] if goal.attack_stage is not None else 0,
                        "credits": np.array([0.0], dtype=np.float32),
                        "ins": 0,
                        "defender": 0
                    }
                elif goal.type == "actor_goal":
                    goal_obs = {
                        "type": MAP_GOAL_TYPES.get(goal.type, 0),
                        "asset_id": 0,
                        "damage": np.array([]),
                        "exposed": (False, False, False),
                        "attack_stage": 0,
                        "credits": goal.credits,
                        "ins": goal.ins,
                        "defender": 0
                    }
                elif goal.type =="defender_not_exceeded_goal":
                    goal_obs = {
                        "type": MAP_GOAL_TYPES.get(goal.type, 0),
                        "asset_id": 0,
                        "damage": np.array([]),
                        "exposed": (False, False, False),
                        "attack_stage": 0,
                        "credits": goal.credits,
                        "ins": goal.ins,
                        "defender": goal.defender
                    }
                goals.append(goal_obs)
            actor_obs["goals"] = tuple(goals)
        if actor.assets is not None:
            actor_obs["assets"] = tuple([asset.id for asset in actor.assets])
        return actor_obs

    def _create_action_obs(self, action: Union[Action,ActionTemplate]):
        action_obs = {
            #"type": MAP_ACTION_TYPES.get(action.type, 0), 
            "card_type": MAP_ACTION_CARD_TYPES.get(action.card_type, 0),
            "target_type": MAP_ACTION_CARD_TYPES.get(action.target_type, 0),
            "actor_type": MAP_ACTION_ACTOR_TYPES.get(action.actor_type, 0),
            "attack_stage": action.attack_stage,
            "oses": tuple(action.oses),
            "asset_categories": tuple(action.asset_categories),
            "impact": np.array(action.impact),
            "effects": tuple(
                [self._create_effect_obs(effect) for effect in action.effects]
            ),
            
            "success_chance": np.array([0.0], dtype=np.float32),
            "detection_chance": np.array([0.0], dtype=np.float32),
            "detection_chance_failed": np.array([0.0], dtype=np.float32),
            "predefined_attack_mask": "",
            "transfer_effects": tuple(),
            "def_type": int(action.def_type) if action.def_type is not None else 0,
            "possible_actions": tuple(),
            "requires_attack_mask": int(action.requires_attack_mask)+1 if action.requires_attack_mask is not None else 0,
            "soph_requirement": action.soph_requirement,
        }
        if isinstance(action, Action):
            action_obs["template_id"] = self._get_action_template_id(action.template_id)
        elif isinstance(action, ActionTemplate):
            action_obs["template_id"] = self._get_action_template_id(action.id)
        else:
            raise ValueError(f"Unknon action type: {type(action)}")

        # Optional attributes
        if action.success_chance is not None:
            action_obs["success_chance"] = np.array([action.success_chance], dtype=np.float32)
        if action.detection_chance is not None:
            action_obs["detection_chance"] = np.array([action.detection_chance], dtype=np.float32)
        if action.detection_chance_failed is not None:
            action_obs["detection_chance_failed"] = np.array([action.detection_chance_failed], dtype=np.float32)
        if action.predefined_attack_mask is not None:
            action_obs["predefined_attack_mask"] = action.predefined_attack_mask
        if action.transfer_effects is not None:
            action_obs["transfer_effects"] = tuple(
                self._create_effect_obs(effect) 
                for effect in action.transfer_effects
            )
        if action.def_type is not None:
            action_obs["def_type"] = action.def_type
        
        if action.possible_actions is not None:
            action_obs["possible_actions"] = tuple(
                [self._get_action_template_id(tid) for tid in action.possible_actions]
            )
        return action_obs
    
    def _create_effect_obs(self, effect: Effect):
        effect_obs = {
            "type": MAP_EFFECT_TYPES.get(effect.type, 0),
            "scope": MAP_SCOPE_TYPES.get(effect.scope, 0),
            "owner_id": self._get_actor_id(effect.owner_id) if effect.owner_id is not None else 0,
            "active": 0,
            "attributes": tuple(),
            "equipment": tuple(),
            "num_effects": 0,
            "probability": np.array([0.0], dtype=np.float32),
            "turns": 0,
            "value": np.array([0.0], dtype=np.float32),
            "damage": np.array([0, 0, 0])
        }
        effect_type_id = MAP_EFFECT_TYPES.get(effect.type, 0)
        if effect_type_id in INC_DEC_EFFECTS:
            effect_obs["attributes"] = tuple(effect.attributes)
            effect_obs["value"] = np.array([effect.value], dtype=np.float32)
        elif effect_type_id in PERMANENT_EFFECTS:
            effect_obs["active"] = int(effect.active) if effect.active is not None else 0
            effect_obs["turns"] = effect.turns if effect.turns is not None else -1
        elif effect_type_id == 2:
            effect_obs["num_effects"] = effect.num_effects
        elif effect_type_id == 6:
            effect_obs["probability"] = np.array([effect.probability], dtype=np.float32)
            effect_obs["damage"] = tuple(effect.value)
        elif effect_type_id == 8:
            effect_obs["equipment"] = tuple([
                self._create_equipment_obs(eq) 
                for eq in effect.equipment
            ])
        return effect_obs

    def _create_equipment_obs(self, equipment: Union[Equipment, EquipmentTemplate]):
        equipment_obs = {
            "type": MAP_EQUIPMENT_TYPES.get(equipment.type, 0),
            "effects": tuple(
                [self._create_effect_obs(eff) for eff in equipment.effects]
            ) if equipment.effects is not None else tuple(),
            "transfer_effects": tuple(
                [
                    self._create_effect_obs(eff) 
                    for eff in equipment.transfer_effects
                ]
            ) if equipment.transfer_effects is not None else tuple(),
            "price": np.array([equipment.price], dtype=np.float32),
            "impact": np.array(equipment.impact) if equipment.impact is not None else np.array([0, 0, 0]),
            "possible_actions": tuple(
                [self._get_action_template_id(x) for x in equipment.possible_actions]
            ) if equipment.possible_actions is not None else tuple()
        }
        if isinstance(equipment, Equipment):
            equipment_obs["active"] = int(equipment.active) + 1
        else:
            equipment_obs["active"] = 0
        eq_type_id = MAP_EQUIPMENT_TYPES.get(equipment.type, 0)
        if eq_type_id in ATTACK_EQUIPMENT:
            equipment_obs["actor_type"] = ActorTypes.Attack.value
        elif eq_type_id in DEFENSE_EQUIPMENT:
            equipment_obs["actor_type"] = ActorTypes.Defense.value
        else:
            equipment_obs["actor_type"] = ActorTypes.Null.value
        if eq_type_id in SINGLE_USE_EQUIPMENT:
            equipment_obs["timing_type"] = TimingTypes.Single_Use.value
        elif eq_type_id in PERMANENT_EQUIPMENT:
            equipment_obs["timing_type"] = TimingTypes.Permanent.value
        else:
            equipment_obs["timing_type"] = TimingTypes.Null.value
        if eq_type_id in LOCAL_EQUIPMENT:
            equipment_obs["scope_type"] = ScopeTypes.Local.value
        elif eq_type_id in GLOBAL_EQUIPMENT:
            equipment_obs["scope_type"] = ScopeTypes.Global.value
        else:
            equipment_obs["scope_type"] = ScopeTypes.Null.value
        return equipment_obs
    
    def _create_asset_obs(self, asset: Asset) -> Dict:
        asset_obs = {
            "id": self._get_asset_id(asset.id),
            "category": asset.category,
            "os": asset.os,
            "attack_stage": asset.attack_stage,
            "parent_asset": 0,
            "child_assets": tuple(asset.child_assets),
            "exposed": asset.exposed,
            "damage": np.array(asset.damage),
            "attack_vectors": tuple(asset.attack_vectors) if asset.attack_vectors is not None else tuple(),
            "dependencies": tuple(asset.dependencies) if asset.dependencies is not None else tuple(),
            "active_exploits": tuple(
                [self._create_equipment_obs(eq) 
                for eq in asset.active_exploits]
            ),
            "permanent_effects": tuple(
                [self._create_effect_obs(effect) 
                for effect in asset.permanent_effects]
            ),
            "played_actions": tuple(
                [self._create_action_obs(action) 
                for action in asset.played_actions]
            ),
            "shield": int(asset.shield)+1 if asset.shield is not None else 0
        }
        if asset.parent_asset is not None:
            asset_obs["parent_asset"] = asset.parent_asset
        return asset_obs