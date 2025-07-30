# Balatro Feature Encoder Specification

## Problem
Current LLM-based text encoding is computationally expensive and inconsistent for structured game data.

## Solution
Single transformer encoder for all cards with type embeddings + structured pointer network.

## Architecture

### Single Transformer Encoder
```python
class BalatroFeatureEncoder(nn.Module):
    def __init__(self, card_dim=64, hidden_dim=128):
        # Card feature embeddings
        self.rank_embed = nn.Embedding(13, card_dim//4)
        self.suit_embed = nn.Embedding(4, card_dim//4)  
        self.enhancement_embed = nn.Embedding(20, card_dim//4)
        self.seal_embed = nn.Embedding(6, card_dim//4)
        
        # Card type embeddings (distinguish hand/joker/shop/etc)
        self.card_type_embed = nn.Embedding(6, card_dim//4)
        
        # Single transformer for all cards
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=card_dim, nhead=8, dim_feedforward=hidden_dim, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        # Component-specific pooling
        self.component_pooling = nn.ModuleDict({
            'hand': nn.Linear(card_dim, hidden_dim),
            'joker': nn.Linear(card_dim, hidden_dim),
            'shop': nn.Linear(card_dim, hidden_dim),
        })
        
    def forward(self, observation):
        all_cards, card_types, type_masks = self.encode_all_cards(observation)
        
        if all_cards.size(0) > 0:
            attended_cards = self.transformer(all_cards)  # Cross-component attention
        else:
            attended_cards = torch.empty(0, self.card_dim)
            
        # Pool by component type for state representation
        hand_embed = self.pool_by_mask(attended_cards, type_masks['hand'])
        joker_embed = self.pool_by_mask(attended_cards, type_masks['joker'])
        shop_embed = self.pool_by_mask(attended_cards, type_masks['shop'])
        
        return {
            'state_embed': torch.cat([hand_embed, joker_embed, shop_embed, game_embed]),
            'card_embeddings': attended_cards,  # For pointer network
            'card_types': card_types,
            'type_masks': type_masks
        }
```

### Integrated Pointer Network
```python
class Agent(nn.Module):
    def get_action_and_value(self, observation, lstm_hidden, ...):
        # Get both state and individual card embeddings
        encoder_output = self.feature_encoder(observation)
        
        # LSTM processes state representation
        x = encoder_output['state_embed'].unsqueeze(1)
        x, new_lstm_hidden = self.torso(x, lstm_hidden)
        
        # Action selection from state
        action_logits = self.actor(x)
        
        # POINTER NETWORK: Select from context-aware card embeddings
        if encoder_output['card_embeddings'].size(0) > 0:
            pointer_embed = self.pointer(x)  # Query vector
            card_embeds = encoder_output['card_embeddings']  # Key vectors
            
            pointer_logits = torch.matmul(pointer_embed, card_embeds.T)
            card_dist = Categorical(logits=pointer_logits)
            card = card_dist.sample()
        else:
            card = torch.zeros(1).to(x.device)
            
        return action, card, ...
```

## Architecture Diagram

```
Observation:
┌─────────────┬─────────────┬─────────────┐
│ Hand Cards  │   Jokers    │ Shop Items  │
│ [K♠, A♥]    │   [Odd Todd] │ [Steel 3♣]  │
└─────────────┴─────────────┴─────────────┘
       │              │              │
       ▼              ▼              ▼
┌─────────────┬─────────────┬─────────────┐
│Card Encoder │Card Encoder │Card Encoder │
│+ Type(hand) │+ Type(joker)│+ Type(shop) │
└─────────────┴─────────────┴─────────────┘
       │              │              │
       └──────────────┼──────────────┘
                      ▼
              ┌───────────────┐
              │ Transformer   │◄─── All cards attend to each other
              │   Encoder     │     (hand ↔ jokers ↔ shop)
              │(no pos encode)│
              └───────┬───────┘
                      │
              ┌───────▼───────┐
              │  Split Output │
              └───────┬───────┘
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
┌─────────────┐ ┌──────────┐ ┌─────────────┐
│ Pool by     │ │Individual│ │ Game State  │
│ Type Masks  │ │   Card   │ │   Encoder   │
│             │ │Embeddings│ │             │
└──────┬──────┘ └─────┬────┘ └──────┬──────┘
       │              │             │
       ▼              │             ▼
┌─────────────┐       │      ┌─────────────┐
│State Embed  │       │      │             │
│[hand,joker, │       │      │             │
│ shop,game]  │       │      │             │
└──────┬──────┘       │      │             │
       │              │      │             │
       ▼              │      ▼             │
┌─────────────┐       │ ┌─────────────┐    │
│    LSTM     │       │ │ Pointer Net │    │
│             │       │ │   Query     │    │
└──────┬──────┘       │ └──────┬──────┘    │
       │              │        │           │
       ├──────────────┘        │           │
       │                       │           │
       ▼                       ▼           │
┌─────────────┐         ┌─────────────┐    │
│   Actor     │         │   Pointer   │    │
│ (Action)    │         │ (Card Select)│   │
└─────────────┘         └─────────────┘    │
                                           │
                               ┌───────────┘
                               ▼
                        ┌─────────────┐
                        │   Critic    │
                        │  (Value)    │
                        └─────────────┘
```

## Benefits

1. **Cross-component attention**: Hand cards attend to jokers for synergy detection
2. **Consistent representations**: Same encoder for state and pointer targets  
3. **Context-aware selection**: Pointer network selects from cards that have "seen" the full game state
4. **Simpler architecture**: One transformer instead of separate encoders
5. **10-100x faster**: No LLM text processing

## Implementation Plan

1. **Feature extraction**: Parse text observations into structured card data
2. **Single encoder**: Implement unified transformer with type embeddings
3. **Dual output**: Return both pooled state + individual card embeddings  
4. **Integration**: Replace `Agent.text_encode()` and update pointer network

## Integration Points
- Replace `Agent.text_encode()` method
- Update `BalatroGymEnv._get_observation()` preprocessing  
- Modify pointer network to use structured embeddings
- Add structured feature extraction to controller.py