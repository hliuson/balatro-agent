#!/usr/bin/env python3
"""Simple test script for BalatroGymEnv"""

import numpy as np
from balatro_env import BalatroGymEnv

def test_basic_functionality():
    """Test basic environment functionality"""
    print("Creating BalatroGymEnv...")
    env = BalatroGymEnv()
    
    print("Testing reset...")
    obs, info = env.reset()
    print(f"Initial observation keys: {list(obs.keys())}")
    print(f"Game state: {obs['game_state_text']}")
    print(f"Hand cards: {len(obs['hand_cards'])} cards")
    print(f"Info: {info}")
    
    print("\nTesting a few random actions...")
    for i in range(5):
        # Sample random action
        action = env.action_space.sample()
        print(f"\nStep {i+1}:")
        print(f"Action: {action}")
        
        obs, reward, done, truncated, info = env.step(action)
        print(f"Reward: {reward}")
        print(f"Done: {done}")
        print(f"Game state: {obs['game_state_text']}")
        
        if done:
            print("Episode finished!")
            break
    
    print("\nClosing environment...")
    env.close()
    print("Test completed!")

if __name__ == "__main__":
    test_basic_functionality()