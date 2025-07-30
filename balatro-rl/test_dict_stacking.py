#!/usr/bin/env python3
"""Test the nested dict stacking/unstacking functions."""

import numpy as np
from ppo import stack_dict_observations, unstack_dict_observations

def test_nested_dict_stacking():
    """Test that nested dict stacking works correctly."""
    print("Testing nested dict stacking...")
    
    # Create sample observations like what we'd get from vectorized environments
    obs_list = [
        {
            "raw_game_state": {
                "hand": [{"value": "5", "suit": "Hearts"}],
                "jokers": [{"name": "Joker"}],
                "game": {"ante": 1, "round": 1, "dollars": 4}
            },
            "action_mask": np.array([1, 0, 1, 0])
        },
        {
            "raw_game_state": {
                "hand": [{"value": "3", "suit": "Spades"}],
                "jokers": [{"name": "Blueprint"}],
                "game": {"ante": 1, "round": 2, "dollars": 8}
            },
            "action_mask": np.array([0, 1, 1, 1])
        }
    ]
    
    # Stack the observations
    stacked = stack_dict_observations(obs_list)
    
    print("Stacked structure:")
    print(f"Keys: {list(stacked.keys())}")
    print(f"raw_game_state keys: {list(stacked['raw_game_state'].keys())}")
    print(f"hand: {stacked['raw_game_state']['hand']}")
    print(f"game: {stacked['raw_game_state']['game']}")
    print(f"action_mask shape: {stacked['action_mask'].shape}")
    
    # Unstack the observations
    unstacked = unstack_dict_observations(stacked, 2)
    
    print("\nUnstacked observations:")
    for i, obs in enumerate(unstacked):
        print(f"Env {i}:")
        print(f"  hand: {obs['raw_game_state']['hand']}")
        print(f"  game: {obs['raw_game_state']['game']}")
        print(f"  action_mask: {obs['action_mask']}")
    
    # Verify round-trip consistency
    assert len(unstacked) == len(obs_list)
    for i, (original, recovered) in enumerate(zip(obs_list, unstacked)):
        print(f"\nVerifying env {i}...")
        assert original['raw_game_state']['game']['ante'] == recovered['raw_game_state']['game']['ante']
        assert original['raw_game_state']['game']['round'] == recovered['raw_game_state']['game']['round']
        assert original['raw_game_state']['game']['dollars'] == recovered['raw_game_state']['game']['dollars']
        assert np.array_equal(original['action_mask'], recovered['action_mask'])
    
    print("✓ All round-trip tests passed!")

if __name__ == "__main__":
    test_nested_dict_stacking()