import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, List, Any, Tuple, Optional
from enum import Enum

# Import the existing controller and formatting functions
import sys
import os
from controller import (
    TrainingBalatroController, 
    Actions, 
    format_card, 
    format_jokers, 
    format_consumables, 
    format_shop_cards, 
    format_boosters, 
    format_vouchers,
    format_game_state
)

class CardSources(Enum):
    HAND = 0
    JOKER = 1
    CONSUMABLE = 2
    SHOP = 3
    BOOSTERPACK = 4
    BOOSTERCARD = 5
    VOUCHER = 6

ACTION_TO_SOURCE = {
    Actions.SELECT_HAND_CARD: CardSources.HAND,
    Actions.SELL_JOKER: CardSources.JOKER,
    Actions.USE_CONSUMABLE: CardSources.CONSUMABLE,
    Actions.BUY_CARD: CardSources.SHOP,
    Actions.BUY_VOUCHER: CardSources.VOUCHER,
    Actions.BUY_BOOSTER: CardSources.BOOSTERPACK,
    Actions.SELECT_BOOSTER_CARD: CardSources.BOOSTERCARD,
}

class BalatroGymEnv(gym.Env):
    """Minimal Balatro Gymnasium Environment using text observations"""
    
    def __init__(self):
        super().__init__()
        
        # Initialize the Balatro controller
        self.controller = TrainingBalatroController(verbose=False)
        self.controller.run_until_policy()
        
        # Define observation space for raw game state
        self.observation_space = spaces.Dict({
            "raw_game_state": spaces.Dict({}),  # Raw dict from controller - flexible structure
            "action_mask": spaces.Box(low=0, high=1, shape=(len(Actions),), dtype=np.int8),
        })
        
        # Define action space
        self.action_space = spaces.Dict({
            "action_type": spaces.Discrete(len(Actions)),
            "card_index": spaces.Box(low=0, high=np.inf, shape=(2,), dtype=np.int32),
            # card_index format: [CardSources.value, index_in_source]
        })
        
        # Track previous ante and round for reward calculation
        self.prev_ante = 1
        self.prev_round = 1
        
        # Track episode statistics
        self.episode_reward = 0.0
        self.episode_length = 0
        self.failed_actions = 0
        self.total_actions = 0

        self.render_mode = "rgb_array"  # Use RGB array for rendering
        
    def reset(self, seed=None, options=None):
        """Reset the environment to start a new episode"""
        super().reset(seed=seed)
        
        # Start a new game using controller's restart_run method
        game_state = self.controller.restart_run()
        self.prev_ante = 1
        self.prev_round = 1
        self.prev_chips = 0
        
        # Reset episode tracking
        self.episode_reward = 0.0
        self.episode_length = 0
        self.failed_actions = 0
        self.total_actions = 0
        
        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action):
        """Execute one step in the environment"""
        action_type = action["action_type"]
        card_index = action["card_index"]
        
        # Convert action to controller format
        controller_action = Actions(action_type)
        
        # Build action list for controller
        try:
            if self._action_needs_card_index(controller_action):
                # Convert 2D card index to appropriate argument
                card_source = CardSources(card_index[0])  # Use enum for validation
                if card_source not in ACTION_TO_SOURCE.values():
                    #THIS should never happen since we use pointer networks.
                    raise ValueError(f"Invalid card source: {card_source}")

                if ACTION_TO_SOURCE[controller_action] != card_source:
                    # This might happen if we don't properly action mask cards.
                    # Ideally the policy should just learn to select the right cards though.
                    raise ValueError(f"Action {controller_action} does not match card source {card_source}")
            
                card_index_in_source = int(card_index[1])
                action_list = [controller_action, card_index_in_source]
            else:
                # Action doesn't need card selection
                action_list = [controller_action]
            
            # Execute action through controller's do_policy_action
            action_valid, game_state = self.controller.do_policy_action(action_list)
            
        except Exception as e:
            action_valid = False
            game_state = self.controller.G  # Get current game state
        
        # Get current state
        obs = self._get_observation()
        info = self._get_info()
        
        # Calculate reward
        current_ante = self._get_current_ante()
        current_round = self._get_current_round()
        reward = self._calculate_reward(current_ante, current_round, action_valid)
        self.prev_ante = current_ante
        self.prev_round = current_round
        self.prev_chips = self._get_current_chips()
        
        # Update episode tracking
        self.episode_reward += reward
        self.episode_length += 1
        self.total_actions += 1
        if not action_valid:
            self.failed_actions += 1
        
        # Check if episode is done
        done = self._is_done()
        
        return obs, reward, done, False, info

    def _get_observation(self) -> Dict[str, Any]:
        """Get raw game state from controller"""
        game_state = self.controller.G
        if not game_state:
            # Return minimal empty game state
            game_state = {
                "hand": [],
                "jokers": [],
                "consumables": [],
                "shop": {"jokers": [], "boosters": [], "vouchers": []},
                "game": {"round": 1, "ante": 1, "dollars": 0}
            }
        
        return {
            "raw_game_state": game_state,
            "action_mask": self._get_action_mask(game_state)
        }
    
    def _get_action_mask(self, game_state: Dict[str, Any]) -> np.ndarray:
        """Generate action mask based on currently valid actions"""
        mask = np.zeros(len(Actions), dtype=np.int8)
        
        try:
            # Get valid actions from controller
            valid_actions = self.controller.get_valid_actions(game_state)
            
            # Convert valid action names to mask
            for valid_action in valid_actions:
                action_name = valid_action.get("action", "")
                try:
                    action_enum = Actions[action_name]
                    #convert from 1-indexed to 0-indexed
                    mask[action_enum.value-1] = 1
                except (KeyError, AttributeError):
                    # Skip invalid action names
                    continue
                    
        except Exception as e:
            # If we can't get valid actions, allow all actions as fallback
            print(f"Warning: Could not get valid actions, allowing all: {e}")
            mask.fill(1)
            
        return mask
    
    def _action_needs_card_index(self, action: Actions) -> bool:
        """Check if action requires card index"""
        card_actions = {
            Actions.SELECT_HAND_CARD,
            Actions.BUY_CARD,
            Actions.BUY_VOUCHER,
            Actions.SELL_JOKER,
            Actions.USE_CONSUMABLE,
            Actions.BUY_BOOSTER,
            Actions.SELECT_BOOSTER_CARD,
        }
        return action in card_actions
    
    def _calculate_reward(self, current_ante: int, current_round: int, action_valid: bool) -> float:
        """Calculate reward based on scored chips as % of necessary chips to beat the round"""
        reward = 0.0
        
        # Get current chips scored and chips required
        current_chips = self._get_current_chips()
        required_chips = self._get_required_chips()
        chip_progress = current_chips - self.prev_chips
        prev_chip_percent = self.prev_chips / required_chips

        # Calculate chip percentage reward
        if chip_progress > 0: 
            chip_percentage = min(chip_progress / required_chips, 1.0 - prev_chip_percent) #cumulative chip reward over the round cannot exceed 1.0
            reward += chip_percentage  # Reward ranges from 0 to 1 based on progress
        
        # Bonus rewards for progression
        if current_round > self.prev_round:
            reward = 1.0 - prev_chip_percent # Give the rest of the reward if we progressed to the next round
            
        return reward
    
    def _get_current_ante(self) -> int:
        """Get current ante from game state"""
        if self.controller.G and self.controller.G.get("game"):
            return self.controller.G["game"].get("ante", 1)
        return 1
    
    def _get_current_round(self) -> int:
        """Get current round from game state"""
        if self.controller.G and self.controller.G.get("game"):
            return self.controller.G["game"].get("round", 1)
        return 1
    
    def _get_current_chips(self) -> int:
        """Get current chips scored from game state"""
        if self.controller.G.get("game"):
            return self.controller.G["game"]["chips"]
        return 0

    def _get_required_chips(self) -> int:
        if self.controller.G.get("ante"):
            ante = self.controller.G["ante"]
            return ante["blinds"]["current"]["chips"]
        else:
            return 1
    
    def _is_done(self) -> bool:
        """Check if episode is complete (game over)"""
        return self.controller.is_episode_done(self.controller.G)
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional info about current state"""
        info = {
            "ante": self._get_current_ante(),
            "round": self._get_current_round(),
            "valid_action": True,  # Could be enhanced based on last action result
            "failed_actions": self.failed_actions,
            "total_actions": self.total_actions,
            "failed_action_rate": self.failed_actions / max(1, self.total_actions)
        }
        
        info["episode_return"] = self.episode_reward
        
        return info
    
    def close(self):
        """Clean up resources"""
        if hasattr(self.controller, 'close'):
            self.controller.close()
    
    def render(self):
        return self._render_frame()

    def _render_frame(self):
        return self.controller.screenshot_np()
