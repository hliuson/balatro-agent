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
            [1, 2, 3, 4],  # Card 1: suit, rank, edition, seal
            [2, 5, 1, 0],  # Card 2: suit, rank, edition, seal
            [3, 1, 0, 2],  # Card 3: suit, rank, edition, seal
            [0, 0, 0, 0],  # Padding
            [0, 0, 0, 0],  # Padding
        ], dtype=np.int32),
        "card_types": np.array([0, 1, 2, 6, 6], dtype=np.int32),  # 6 is padding
        "type_masks": {
            "hand": np.array([1, 1, 0, 0, 0], dtype=np.int8),
            "joker": np.array([0, 0, 1, 0, 0], dtype=np.int8),
            "shop": np.array([0, 0, 0, 0, 0], dtype=np.int8),
            "consumable": np.array([0, 0, 0, 0, 0], dtype=np.int8),
            "booster": np.array([0, 0, 0, 0, 0], dtype=np.int8),
            "voucher": np.array([0, 0, 0, 0, 0], dtype=np.int8),
        },
        "game_state": np.array([5, 2, 100], dtype=np.int32),  # ante, round, money
    }

def test_feature_encoder():
    """Test the feature encoder with sample data."""
    print("Testing BalatroFeatureEncoder...")
    
    # Create encoder
    encoder = BalatroFeatureEncoder(
        card_dim=64,
        hidden_dim=128,
        num_transformer_layers=2,
        num_attention_heads=4
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
    print(f"  - Component embeddings: {len(output['component_embeddings'])}")
    
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
    
    # Create mock game state
    game_state = {
        "hand": [
            {"suit": 1, "rank": 5, "edition": 0, "seal": 0},
            {"suit": 2, "rank": 3, "edition": 1, "seal": 0},
        ],
        "jokers": [
            {"id": 1, "edition": 0, "seal": 0},
        ],
        "game": {
            "ante": 3,
            "round": 2,
            "money": 50
        }
    }
    
    # Extract features
    features = CardFeatureExtractor.extract_features(game_state)
    
    print("✓ Card feature extraction works")
    print(f"  - Cards shape: {features['cards'].shape}")
    print(f"  - Card types: {features['card_types']}")
    print(f"  - Game state: {features['game_state']}")
    
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
