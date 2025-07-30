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
from simple_curriculum import SimpleCurriculumEnv

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

class BalatroGymEnv(SimpleCurriculumEnv, gym.Env):
    """Minimal Balatro Gymnasium Environment using text observations"""
    
    def __init__(self):
        # Initialize SimpleCurriculumEnv first
        SimpleCurriculumEnv.__init__(self)
        gym.Env.__init__(self)
        
        # Initialize the Balatro controller
        self.controller = TrainingBalatroController(verbose=False)
        self.controller.run_until_policy()
        
        # Define observation space
        self.observation_space = spaces.Dict({
            "game_state_text": spaces.Text(max_length=2048),
            "hand_cards": spaces.Sequence(spaces.Text(max_length=256)),
            "jokers": spaces.Sequence(spaces.Text(max_length=256)),
            "consumables": spaces.Sequence(spaces.Text(max_length=256)),
            "shop_items": spaces.Sequence(spaces.Text(max_length=256)),
            "boosters": spaces.Sequence(spaces.Text(max_length=256)),
            "vouchers": spaces.Sequence(spaces.Text(max_length=256)),
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
        self.prev_chips = 0
        self.prev_state = None
        
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
        self.prev_state = None
        
        # Reset episode tracking
        self.episode_reward = 0.0
        self.episode_length = 0
        self.failed_actions = 0
        self.total_actions = 0
        
        # Reset curriculum rewards
        self.reset_curriculum()
        
        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action):
        """Execute one step in the environment"""
        # Update curriculum state tracking before taking action
        if self.controller.G:
            self.update_curriculum_state(self.controller.G)
        
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
        
        # Calculate reward using curriculum manager
        current_ante = self._get_current_ante()
        current_round = self._get_current_round()
        current_chips = self._get_current_chips()
        
        episode_info = {
            'ante': current_ante,
            'round': current_round,
            'prev_ante': self.prev_ante,
            'prev_round': self.prev_round,
            'chips': current_chips,
            'prev_chips': self.prev_chips,
            'episode_length': self.episode_length,
            'total_actions': self.total_actions
        }
        
        # Use curriculum reward system
        current_state = self.controller.G
        reward, reward_breakdown = self.calculate_curriculum_reward(
            action=action,
            current_state=current_state,
            prev_state=self.prev_state or current_state,
            episode_info=episode_info
        )
        self._last_reward_breakdown = reward_breakdown
        
        # Update tracking
        self.prev_ante = current_ante
        self.prev_round = current_round
        self.prev_chips = current_chips
        self.prev_state = current_state.copy() if current_state else None
        
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
        """Get current observation from game state with individual card strings for pointer network"""
        game_state = self.controller.G
        if not game_state:
            # Return empty observation if no game state
            return {
                "game_state_text": "",
                "hand_cards": [],
                "jokers": [],
                "consumables": [],
                "shop_items": [],
                "boosters": [],
                "vouchers": [],
                "action_mask": np.ones(len(Actions), dtype=np.int8)  # Allow all actions if no state
            }
        
        # Get overall game state text
        game_state_text = format_game_state(game_state)
        
        # Get individual card strings for pointer network
        hand_cards = []
        if game_state.get("hand"):
            for card in game_state["hand"]:
                hand_cards.append(format_card(card))
        
        # Use the refactored formatting functions that return arrays
        jokers = format_jokers(game_state.get("jokers", []))
        consumables = format_consumables(game_state.get("consumables", []))
        
        # Shop items, boosters, vouchers
        shop_items = []
        boosters = []
        vouchers = []
        
        if game_state.get("shop"):
            shop = game_state["shop"]
            
            # Shop cards (contains mixed types) - use refactored function
            shop_items = format_shop_cards(shop.get("jokers", []))
            
            # Boosters - use refactored function
            boosters = format_boosters(shop.get("boosters", []))
            
            # Vouchers - use refactored function
            vouchers = format_vouchers(shop.get("vouchers", []))
        
        # Generate action mask based on valid actions
        action_mask = self._get_action_mask(game_state)
        
        return {
            "game_state_text": game_state_text,
            "hand_cards": hand_cards,
            "jokers": jokers,
            "consumables": consumables,
            "shop_items": shop_items,
            "boosters": boosters,
            "vouchers": vouchers,
            "action_mask": action_mask
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
        """Legacy reward calculation - now handled by curriculum reward manager"""
        # This method is kept for backward compatibility but should not be used
        # All reward calculation is now done through self.reward_manager
        return 0.0
    
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
        info["reward_breakdown"] = getattr(self, '_last_reward_breakdown', {})
        info["reward_status"] = self.get_curriculum_status()
        
        return info
    
    def close(self):
        """Clean up resources"""
        if hasattr(self.controller, 'close'):
            self.controller.close()
    
    def render(self):
        return self._render_frame()

    def _render_frame(self):
        return self.controller.screenshot_np()
    
    # Curriculum learning methods (for backward compatibility)
    def enable_reward(self, name: str):
        """Enable a specific reward function"""
        if name in self.rewards:
            self.rewards[name].enabled = True
    
    def disable_reward(self, name: str):
        """Disable a specific reward function"""
        if name in self.rewards:
            self.rewards[name].enabled = False
    
    def set_reward_weight(self, name: str, weight: float):
        """Set weight for a specific reward function"""
        if name in self.rewards:
            self.rewards[name].weight = weight
    
    def get_reward_status(self):
        """Get status of all reward functions"""
        return self.get_curriculum_status()
    
    def get_reward_breakdown(self):
        """Get breakdown of last reward calculation"""
        return getattr(self, '_last_reward_breakdown', {})
