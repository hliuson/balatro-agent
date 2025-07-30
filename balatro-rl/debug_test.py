#!/usr/bin/env python3
"""Debug test to check feature encoder keys."""

import torch
from feature_encoder import BalatroFeatureEncoder, CardFeatureExtractor

# Test 1: Check what CardFeatureExtractor returns
print("=== Test 1: CardFeatureExtractor output ===")
game_state = {
    "hand": [
        {"value": "5", "suit": "Hearts", "ability_name": "Default Base", "seal": "none"},
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

features = CardFeatureExtractor.extract_features(game_state)
print("Keys:", list(features.keys()))
print("Cards:", features['cards'])
print("Card types:", features['card_types'])
print("Type masks keys:", list(features['type_masks'].keys()))
print("Game state:", features['game_state'])

# Test 2: Convert to tensors and test encoder
print("\n=== Test 2: Tensor conversion ===")
cards = torch.tensor(features['cards']).unsqueeze(0)
card_types = torch.tensor(features['card_types']).unsqueeze(0)
game_state_tensor = torch.tensor(features['game_state']).unsqueeze(0)

type_masks = {}
for key, mask in features['type_masks'].items():
    type_masks[key] = torch.tensor(mask, dtype=torch.bool).unsqueeze(0)

print(f"Cards shape: {cards.shape}")
print(f"Card types shape: {card_types.shape}")
print(f"Game state shape: {game_state_tensor.shape}")

# Test 3: Check encoder input
print("\n=== Test 3: Encoder input ===")
observation = {
    'cards': cards,
    'card_types': card_types,
    'type_masks': type_masks,
    'game_state': game_state_tensor
}

print("Observation keys:", list(observation.keys()))
print("Cards key exists:", 'cards' in observation)

# Test 4: Create encoder and forward pass
print("\n=== Test 4: Encoder forward pass ===")
encoder = BalatroFeatureEncoder(card_dim=84, hidden_dim=128)

try:
    with torch.no_grad():
        output = encoder(observation)
    print("✓ Success!")
    print(f"Output keys: {list(output.keys())}")
    print(f"State embed shape: {output['state_embed'].shape}")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()