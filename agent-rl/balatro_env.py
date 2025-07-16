"""
Balatro Environment for TorchRL.

This module provides a TorchRL environment wrapper around the TrainingBalatroController.
"""

import torch
from typing import Optional, Dict, Any
from tensordict import TensorDict
from torchrl.envs import EnvBase
import torchrl.data
from torchrl.data.tensor_specs import DiscreteTensorSpec

from controller import TrainingBalatroController, State, Actions, format_game_state


class BalatroEnv(EnvBase):
    """
    TorchRL environment wrapper for Balatro.
    
    This environment wraps the TrainingBalatroController to provide a standard
    TorchRL interface for training RL agents on Balatro.
    """
    
    def __init__(self, controller: Optional[TrainingBalatroController] = None, 
                 device: str = "cpu", verbose: bool = False):
        """
        Initialize the Balatro environment.
        
        Args:
            controller: TrainingBalatroController instance. If None, creates a new one.
            device: Device to place tensors on
            verbose: Whether to print debug information
        """
        self.verbose = verbose
        
        # Initialize controller
        if controller is None:
            controller = TrainingBalatroController(verbose=verbose)
        self.controller = controller
        
        # Call parent constructor
        super().__init__(device=device)
        
        # Current game state
        self._current_state = None
        
        # Episode tracking
        self._episode_step = 0
        self._episode_reward = 0.0
        
        """Define the environment specifications."""
        # Observation spec: text string using NonTensor
        self.observation_spec = torchrl.data.NonTensor()
        
        # Action spec: discrete actions (21 possible actions)
        self.action_spec = torchrl.data.Categorical(
            n=len(Actions), 
            device=self.device, 
            dtype=torch.int64
        )
        
        # Reward spec: scalar reward
        self.reward_spec = torchrl.data.Unbounded(
            shape=(1, 1), 
            device=self.device, 
            dtype=torch.float32
        )
    
    def _reset(self, tensordict: TensorDict, **kwargs) -> TensorDict:
        """
        Reset the environment to start a new episode.
        
        Args:
            tensordict: Input tensordict (unused for reset)
            
        Returns:
            TensorDict with initial observation
        """
        if self.verbose:
            print("Resetting Balatro environment")
        
        # Reset controller and get initial state
        self.controller.reset_episode()
        self._current_state = self.controller.run_until_policy()
        
        # Reset episode tracking
        self._episode_step = 0
        self._episode_reward = 0.0
        
        # Format observation
        obs_text = format_game_state(self._current_state)
        valid_actions = self.controller.get_valid_actions(self._current_state)
        # Create output tensordict
        out = TensorDict(
            {
                "observation": [obs_text],  # NonTensor expects a list
                "valid_actions": valid_actions,  # Pass valid actions directly
                "done": torch.tensor([False], dtype=torch.bool, device=self.device),
                "terminated": torch.tensor([False], dtype=torch.bool, device=self.device),
                "truncated": torch.tensor([False], dtype=torch.bool, device=self.device),
                "is_init": torch.tensor([True], dtype=torch.bool, device=self.device),  # LSTM needs this
            },
            batch_size=self.batch_size,
            device=self.device
        )
        
        if self.verbose:
            print(f"Reset complete. Initial state: {State(self._current_state['state']).name}")
            print(f"Observation length: {len(obs_text)} characters")
        
        return out
    
    def _step(self, tensordict: TensorDict) -> TensorDict:
        """
        Execute one step in the environment.
        
        Args:
            tensordict: Input tensordict containing action
            
        Returns:
            TensorDict with next observation, reward, done flags
        """
        # Extract action
        action_idx = tensordict["action"].item()
        
        if self.verbose:
            print(f"Step {self._episode_step}: Action index {action_idx}")
        
        # Convert action index to Actions enum
        try:
            action_enum = list(Actions)[action_idx]
        except IndexError:
            raise ValueError(f"Invalid action index: {action_idx}. Must be 0-{len(Actions)-1}")
        
        if self.verbose:
            print(f"Action: {action_enum.name}")
        
        # Get valid actions to find the action spec
        valid_actions = self.controller.get_valid_actions(self._current_state)
        
        # Find matching action spec
        action_spec = None
        reward = 0.0
        for spec in valid_actions:
            if Actions[spec["action"]] == action_enum:
                action_spec = spec
                break
        
        if action_spec is None:
            if self.verbose:
                print(f"Action {action_enum.name} not valid. Available: {[Actions[s['action']].name for s in valid_actions]}")
            # Return current state with small negative reward
            reward = -0.01
            done = False
            terminated = False
            truncated = False
            next_state = self._current_state
        else:
            # Build complete action with parameters using existing logic
            if tensordict.get("selected_cards") is not None:
                complete_action = [action_spec["action"], tensordict["selected_cards"]]
            else:
                complete_action = [action_spec["action"]]
            
            if self.verbose:
                print(f"Complete action: {complete_action}")
            
            # Execute action
            try:
                next_state = self.controller.do_policy_action(complete_action)
                self._current_state = next_state
                
                # Calculate reward and done flags
                done = self.controller.is_episode_done(next_state)
                terminated = done
                truncated = False
                
                # Get reward
                step_reward = 0.0  # Small step reward
                if done:
                    episode_reward = self.controller.get_episode_reward(next_state)
                    reward = episode_reward
                    if self.verbose:
                        print(f"Episode finished! Final reward: {reward}")
                else:
                    reward = step_reward
                
            except Exception as e:
                if self.verbose:
                    print(f"Error executing action: {e}")
                # Penalize invalid actions
                reward = -0.1
                done = False
                terminated = False
                truncated = False
                next_state = self._current_state
        
        # Update episode tracking
        self._episode_step += 1
        self._episode_reward += reward
        
        # Format next observation
        obs_text = format_game_state(next_state)
        valid_actions = self.controller.get_valid_actions(next_state)
        # Create output tensordict
        out = TensorDict(
            {
                "done": torch.tensor([done], dtype=torch.bool, device=self.device),
                "terminated": torch.tensor([terminated], dtype=torch.bool, device=self.device),
                "truncated": torch.tensor([truncated], dtype=torch.bool, device=self.device),
                "valid_actions": valid_actions,
                "next": TensorDict({
                    "reward": torch.tensor([reward], dtype=torch.float32, device=self.device),
                    "observation": [obs_text],
                }, batch_size=self.batch_size, device=self.device)
            },
            batch_size=self.batch_size,
            device=self.device
        )
        
        if self.verbose:
            print(f"Step complete. Reward: {reward}, Done: {done}")
            if not done:
                print(f"Next state: {State(next_state['state']).name}")
        
        return out
    
    def _set_seed(self, seed: Optional[int]):
        """Set the random seed for reproducibility."""
        if seed is not None:
            torch.manual_seed(seed)
            # Note: controller uses its own random seed management
        return seed
    
    def close(self):
        """Clean up resources."""
        if hasattr(self, 'controller') and self.controller:
            self.controller.close()
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.close()


def make_balatro_env(verbose: bool = False, device: str = "cpu") -> BalatroEnv:
    """
    Factory function to create a Balatro environment.
    
    Args:
        verbose: Whether to enable verbose logging
        device: Device to place tensors on
        
    Returns:
        BalatroEnv instance
    """
    return BalatroEnv(verbose=verbose, device=device)


if __name__ == "__main__":
    # Test the environment
    print("Testing BalatroEnv...")
    
    # Create environment
    env = make_balatro_env(verbose=True)
    
    try:
        # Test reset
        print("\n=== Testing Reset ===")
        reset_data = env.reset()
        print("Reset data keys:", list(reset_data.keys()))
        print("Observation type:", type(reset_data["observation"]))
        print("Observation sample:", reset_data["observation"][0][:100] + "...")
        
        print("Action spec:", env.action_spec)
        # Test random steps
        print("\n=== Testing Random Steps ===")
        for i in range(3):
            # Sample random action
            action = env.action_spec.rand()
            print(f"\nStep {i+1}: Action = {action}")
            
            # Create action tensordict
            action_td = TensorDict(
                {"action": action},
                batch_size=env.batch_size,
                device=env.device
            )
            
            # Step
            step_data = env.step(action_td)
            print("Step data keys:", list(step_data.keys()))
            print("Reward:", step_data["reward"].item())
            print("Done:", step_data["done"].item())
            
            if step_data["done"].item():
                print("Episode finished!")
                break
        
        # Test rollout
        print("\n=== Testing Rollout ===")
        rollout = env.rollout(max_steps=5)
        print("Rollout shape:", rollout.batch_size)
        print("Rollout keys:", list(rollout.keys()))
        print("Total reward:", rollout["reward"].sum().item())
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()
    
    print("BalatroEnv testing complete!")