"""
Simplified self-managing curriculum rewards
"""
from abc import ABC, abstractmethod
from typing import Dict, Any
from collections import deque
import numpy as np

class SelfManagedReward(ABC):
    """Base class for rewards that manage their own curriculum progression"""
    
    def __init__(self, 
                 initial_weight: float = 0.0,
                 enabled: bool = False,
                 success_threshold: float = 0.8,
                 window_size: int = 100,
                 phase_out_steps: int = 1000):
        self.weight = initial_weight
        self.enabled = enabled
        self.success_threshold = success_threshold
        self.window_size = window_size
        self.phase_out_steps = phase_out_steps
        
        # Self-management state
        self.success_history = deque(maxlen=window_size)
        self.is_phasing_out = False
        self.phase_out_remaining = 0
        self.original_weight = initial_weight
        self.last_reward = 0.0
        
    @abstractmethod
    def calculate_raw_reward(self, action, current_state, prev_state, episode_info) -> float:
        """Calculate the raw reward (0 or 1 typically)"""
        pass
    
    def update_state(self, current_state: Dict[str, Any]):
        """Update internal tracking state"""
        pass
    
    def get_weighted_reward(self, *args, **kwargs) -> float:
        """Get reward and update progression"""
        if not self.enabled:
            return 0.0
            
        raw_reward = self.calculate_raw_reward(*args, **kwargs)
        
        # Track success for curriculum progression
        self.success_history.append(1.0 if raw_reward > 0 else 0.0)
        
        # Check if we should start phasing out
        if not self.is_phasing_out and len(self.success_history) >= self.window_size // 2:
            success_rate = np.mean(self.success_history)
            if success_rate >= self.success_threshold:
                self.start_phase_out()
        
        # Update phase-out if active
        if self.is_phasing_out:
            self.update_phase_out()
        
        self.last_reward = raw_reward * self.weight
        return self.last_reward
    
    def start_phase_out(self):
        """Begin phasing out this reward"""
        self.is_phasing_out = True
        self.phase_out_remaining = self.phase_out_steps
        self.original_weight = self.weight
        print(f"🎓 {self.__class__.__name__} mastered! Starting phase-out (success rate: {np.mean(self.success_history):.2f})")
    
    def update_phase_out(self):
        """Update phase-out progress"""
        if self.phase_out_remaining <= 0:
            self.weight = 0.0
            self.enabled = False
            print(f"✅ {self.__class__.__name__} phase-out complete")
        else:
            # Linear decay
            progress = 1.0 - (self.phase_out_remaining / self.phase_out_steps)
            self.weight = self.original_weight * (1.0 - progress)
            self.phase_out_remaining -= 1
    
    def reset(self):
        """Reset for new episode"""
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status"""
        return {
            'enabled': self.enabled,
            'weight': self.weight,
            'success_rate': np.mean(self.success_history) if self.success_history else 0.0,
            'is_phasing_out': self.is_phasing_out,
            'phase_out_progress': 1.0 - (self.phase_out_remaining / self.phase_out_steps) if self.is_phasing_out else 0.0
        }


class MainReward(SelfManagedReward):
    """Main chip-based reward (never phases out)"""
    
    def __init__(self):
        super().__init__(initial_weight=1.0, enabled=True, success_threshold=float('inf'))  # Never phase out
        self.prev_chips = 0
        self.prev_chip_percent = 0.0
    
    def calculate_raw_reward(self, action, current_state, prev_state, episode_info) -> float:
        """Original chip-based reward logic"""
        reward = 0.0
        
        current_chips = current_state.get("game", {}).get("chips", 0)
        required_chips = current_state.get("ante", {}).get("blinds", {}).get("current", {}).get("chips", 1)
        
        if required_chips <= 0:
            return 0.0
            
        chip_progress = current_chips - self.prev_chips
        
        if chip_progress > 0: 
            chip_percentage = min(chip_progress / required_chips, 1.0 - self.prev_chip_percent)
            reward += chip_percentage
        
        # Bonus for round progression
        current_round = episode_info.get('round', 1)
        prev_round = episode_info.get('prev_round', 1)
        if current_round > prev_round:
            reward = 1.0 - self.prev_chip_percent
            
        # Update tracking
        self.prev_chips = current_chips
        self.prev_chip_percent = current_chips / required_chips
        
        return reward
    
    def reset(self):
        super().reset()
        self.prev_chips = 0
        self.prev_chip_percent = 0.0


class CardCountReward(SelfManagedReward):
    """Reward for playing 5-card hands"""
    
    def __init__(self):
        super().__init__(initial_weight=0.3, enabled=True)
        self.last_hand_size = None
        self.last_hands_left = None
    
    def update_state(self, current_state: Dict[str, Any]):
        """Track hand size and hands remaining"""
        self.last_hand_size = len(current_state.get("hand", []))
        if current_state.get("round"):
            self.last_hands_left = current_state["round"].get("hands_left", 0)
    
    def calculate_raw_reward(self, action, current_state, prev_state, episode_info) -> float:
        """Give reward when a 5-card hand is played"""
        from controller import Actions
        if action.get("action_type") != Actions.PLAY_SELECTED.value:
            return 0.0
        
        current_hand_size = len(current_state.get("hand", []))
        current_hands_left = current_state.get("round", {}).get("hands_left", 0)
        
        # Check if hand was played and 5 cards were used
        if (self.last_hands_left is not None and 
            self.last_hand_size is not None and 
            self.last_hands_left > current_hands_left):
            
            cards_played = self.last_hand_size - current_hand_size
            return 1.0 if cards_played == 5 else 0.0
        
        return 0.0
    
    def reset(self):
        super().reset()
        self.last_hand_size = None
        self.last_hands_left = None


class BigHandReward(SelfManagedReward):
    """Reward for playing good poker hands"""
    
    def __init__(self):
        super().__init__(initial_weight=0.0, enabled=False)  # Enabled when cardcount phases out
        self.target_hands = {'Straight', 'Flush', 'Full House', 'Straight Flush', 'Royal Flush', 'Four of a Kind'}
        self.tracked_hand_info = None
        self.last_hands_left = None
    
    def update_state(self, current_state: Dict[str, Any]):
        """Track hand information before it gets cleared"""
        if current_state.get("round"):
            round_info = current_state["round"]
            
            if round_info.get("current_hand") and round_info["current_hand"].get("handname"):
                self.tracked_hand_info = {
                    "handname": round_info["current_hand"]["handname"],
                    "hands_left": round_info.get("hands_left", 0)
                }
            
            self.last_hands_left = round_info.get("hands_left", 0)
    
    def calculate_raw_reward(self, action, current_state, prev_state, episode_info) -> float:
        """Give reward when target hands are played"""
        from controller import Actions
        if action.get("action_type") != Actions.PLAY_SELECTED.value:
            return 0.0
        
        current_hands_left = current_state.get("round", {}).get("hands_left", 0)
        
        # Check if hand was played
        if self.last_hands_left is not None and self.last_hands_left > current_hands_left:
            if self.tracked_hand_info and self.tracked_hand_info["handname"] in self.target_hands:
                self.tracked_hand_info = None  # Clear after use
                return 1.0
        
        return 0.0
    
    def reset(self):
        super().reset()
        self.tracked_hand_info = None
        self.last_hands_left = None


class SimpleCurriculumEnv:
    """Mixin to add self-managing curriculum to environment"""
    
    def __init__(self):
        # Initialize reward functions
        self.rewards = {
            'main': MainReward(),
            'cardcount': CardCountReward(), 
            'bighand': BigHandReward()
        }
        
        # Set up progression callback
        self._setup_progression()
    
    def _setup_progression(self):
        """Set up curriculum progression chain"""
        # Store reference to self for callbacks
        curriculum_env = self
        
        # Override phase-out methods to trigger next reward
        original_cardcount_phase_out = self.rewards['cardcount'].update_phase_out
        def cardcount_phase_out():
            original_cardcount_phase_out()
            if self.rewards['cardcount'].phase_out_remaining == 0 and not self.rewards['bighand'].enabled:
                curriculum_env._enable_next_reward('bighand')
        self.rewards['cardcount'].update_phase_out = cardcount_phase_out
    
    def _enable_next_reward(self, reward_name: str):
        """Enable next reward when current one phases out"""
        if reward_name in self.rewards:
            self.rewards[reward_name].enabled = True
            self.rewards[reward_name].weight = 0.2
            print(f"🚀 Enabled {reward_name} reward for next curriculum stage")
    
    def update_curriculum_state(self, current_state: Dict[str, Any]):
        """Update all reward tracking states"""
        for reward in self.rewards.values():
            reward.update_state(current_state)
    
    def calculate_curriculum_reward(self, action, current_state, prev_state, episode_info) -> tuple:
        """Calculate total reward and breakdown"""
        total_reward = 0.0
        breakdown = {}
        
        for name, reward_func in self.rewards.items():
            reward_value = reward_func.get_weighted_reward(action, current_state, prev_state, episode_info)
            total_reward += reward_value
            breakdown[name] = reward_value
        
        return total_reward, breakdown
    
    def reset_curriculum(self):
        """Reset all rewards for new episode"""
        for reward in self.rewards.values():
            reward.reset()
    
    def get_curriculum_status(self):
        """Get status of all rewards"""
        return {name: reward.get_status() for name, reward in self.rewards.items()}