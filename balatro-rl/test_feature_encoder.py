#!/usr/bin/env python3
"""
Test script to verify the feature encoder implementation.
"""

import numpy as np
import torch
from feature_encoder import BalatroFeatureEncoder, CardFeatureExtractor

def create_test_observation():
    """Create a test observation with sample data."""
    return {
        "cards": np.array([
            [2, 1, 1, 0, 0, 0],  # Card 1: rank, suit, enhancement, seal, edition, joker_id
            [5, 2, 2, 1, 1, 0],  # Card 2: rank, suit, enhancement, seal, edition, joker_id (foil gold card)
            [0, 0, 0, 0, 2, 1],  # Card 3: joker card with holo edition, joker_id=1
        ], dtype=np.int32),
        "card_types": np.array([0, 0, 1], dtype=np.int32),  # hand, hand, joker
        "type_masks": {
            "hand": np.array([1, 1, 0], dtype=np.int8),
            "joker": np.array([0, 0, 1], dtype=np.int8),
            "shop": np.array([0, 0, 0], dtype=np.int8),
            "consumable": np.array([0, 0, 0], dtype=np.int8),
            "booster": np.array([0, 0, 0], dtype=np.int8),
            "voucher": np.array([0, 0, 0], dtype=np.int8),
        },
        "game_state": np.array([5, 2, 100], dtype=np.int32),  # ante, round, money
    }

def test_feature_encoder():
    """Test the feature encoder with sample data."""
    print("Testing BalatroFeatureEncoder...")
    
    # Create encoder
    encoder = BalatroFeatureEncoder(
        card_dim=84,
        hidden_dim=128,
        num_transformer_layers=2,
        num_attention_heads=12
    )
    
    # Create test observation
    obs = create_test_observation()
    
    # Convert to tensors
    cards = torch.from_numpy(obs["cards"]).unsqueeze(0)  # Add batch dimension
    card_types = torch.from_numpy(obs["card_types"]).unsqueeze(0)
    game_state = torch.from_numpy(obs["game_state"]).unsqueeze(0)
    
    type_masks = {}
    for key, mask in obs["type_masks"].items():
        type_masks[key] = torch.from_numpy(mask).unsqueeze(0)
    
    # Create input dict
    input_dict = {
        "cards": cards,
        "card_types": card_types,
        "type_masks": type_masks,
        "game_state": game_state
    }
    
    # Forward pass
    with torch.no_grad():
        output = encoder(input_dict)
    
    print("✓ Feature encoder created successfully")
    print(f"  - State embedding shape: {output['state_embed'].shape}")
    print(f"  - Card embeddings shape: {output['card_embeddings'].shape}")
    print(f"  - Card types shape: {output['card_types'].shape}")
    print(f"  - Type masks keys: {list(output['type_masks'].keys())}")
    
    # Test with batch
    batch_size = 3
    batch_cards = cards.repeat(batch_size, 1, 1)
    batch_card_types = card_types.repeat(batch_size, 1)
    batch_game_state = game_state.repeat(batch_size, 1)
    
    batch_type_masks = {}
    for key, mask in type_masks.items():
        batch_type_masks[key] = mask.repeat(batch_size, 1)
    
    batch_input = {
        "cards": batch_cards,
        "card_types": batch_card_types,
        "type_masks": batch_type_masks,
        "game_state": batch_game_state
    }
    
    with torch.no_grad():
        batch_output = encoder(batch_input)
    
    print("✓ Batch processing works")
    print(f"  - Batch state embedding shape: {batch_output['state_embed'].shape}")
    print(f"  - Batch card embeddings shape: {batch_output['card_embeddings'].shape}")
    
    return True

def test_card_feature_extractor():
    """Test the card feature extractor."""
    print("\nTesting CardFeatureExtractor...")
    
    # Create mock game state matching actual structure from controllers.py
    game_state = {
        "hand": [
            {"value": "5", "suit": "Hearts", "ability_name": "Default Base", "seal": "none"},
            {"value": "3", "suit": "Spades", "ability_name": "Bonus Card", "seal": "none"},
        ],
        "jokers": [
            {"name": "Joker"},
        ],
        "consumables": [],
        "shop": {"jokers": [], "boosters": [], "vouchers": []},
        "game": {
            "ante": 3,
            "round": 2,
            "dollars": 50
        }
    }
    
    # Extract features
    features = CardFeatureExtractor.extract_features(game_state)
    
    print("✓ Card feature extraction works")
    print(f"  - Cards count: {len(features['cards'])}")
    print(f"  - Card types: {features['card_types']}")
    print(f"  - Game state: {features['game_state']}")
    
    # Convert to tensors for encoder testing
    cards_tensor = torch.tensor(features['cards']).unsqueeze(0)
    card_types_tensor = torch.tensor(features['card_types']).unsqueeze(0)
    game_state_tensor = torch.tensor(features['game_state']).unsqueeze(0)
    
    type_masks_tensor = {}
    for key, mask in features['type_masks'].items():
        type_masks_tensor[key] = torch.tensor(mask).unsqueeze(0)
    
    # Test with the encoder
    encoder = BalatroFeatureEncoder()
    input_dict = {
        "cards": cards_tensor,
        "card_types": card_types_tensor,
        "type_masks": type_masks_tensor,
        "game_state": game_state_tensor
    }
    
    with torch.no_grad():
        output = encoder(input_dict)
    
    print("✓ CardFeatureExtractor works with encoder")
    print(f"  - State embedding shape: {output['state_embed'].shape}")
    print(f"  - Card embeddings shape: {output['card_embeddings'].shape}")
    
    return True

def main():
    """Run all tests."""
    print("=" * 50)
    print("Feature Encoder Test Suite")
    print("=" * 50)
    
    try:
        test_feature_encoder()
        test_card_feature_extractor()
        
        print("\n" + "=" * 50)
        print("✅ All tests passed!")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
