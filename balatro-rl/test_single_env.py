#!/usr/bin/env python3

"""Test script to debug Balatro environment initialization"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'balatro-controllers'))

print("Testing single Balatro environment...")

try:
    from balatro_env import BalatroGymEnv
    print("✓ Successfully imported BalatroGymEnv")
    
    print("Creating environment...")
    env = BalatroGymEnv()
    print("✓ Successfully created environment")
    
    print("Resetting environment...")
    obs, info = env.reset()
    print("✓ Successfully reset environment")
    print(f"Observation keys: {list(obs.keys())}")
    
    print("Closing environment...")
    env.close()
    print("✓ Successfully closed environment")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()