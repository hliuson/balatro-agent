import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
from vocabularies import JOKER_VOCAB, TAROT_VOCAB, PLANET_VOCAB, SPECTRAL_VOCAB, VOCAB_SIZES

# Global vocabularies for consistent mapping
ENHANCEMENT_VOCAB = {
    'Default Base': 0, 'Bonus': 1, 'Mult': 2, 'Wild Card': 3,
    'Glass Card': 4, 'Steel Card': 5, 'Stone Card': 6, 'Gold Card': 7, 'Lucky Card': 8
}

SUIT_VOCAB = {'Spades': 1, 'Hearts': 2, 'Diamonds': 3, 'Clubs': 4}

RANK_VOCAB = {
    'Ace': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
    '8': 8, '9': 9, '10': 10, 'Jack': 11, 'Queen': 12, 'King': 13
}

SEAL_VOCAB = {'none': 0, 'Gold': 1, 'Red': 2, 'Blue': 3, 'Purple': 4}

CONSUMABLE_VOCAB = {**TAROT_VOCAB, **PLANET_VOCAB, **SPECTRAL_VOCAB}

BOOSTER_VOCAB = {
    'Arcana Pack': 1, 'Celestial Pack': 2, 'Spectral Pack': 3,
    'Standard Pack': 4, 'Jumbo Standard Pack': 5, 'Mega Standard Pack': 6
}

VOUCHER_VOCAB = {
    # Base vouchers (tier 1)
    'Overstock': 1, 'Clearance Sale': 2, 'Hone': 3, 'Reroll Surplus': 4, 'Crystal Ball': 5,
    'Telescope': 6, 'Grabber': 7, 'Wasteful': 8, 'Tarot Merchant': 9, 'Planet Merchant': 10,
    'Seed Money': 11, 'Blank': 12, 'Magic Trick': 13, 'Hieroglyph': 14, 'Director\'s Cut': 15,
    'Paint Brush': 16,
    # Upgraded vouchers (tier 2)
    'Overstock Plus': 17, 'Liquidation': 18, 'Glow Up': 19, 'Reroll Glut': 20, 'Omen Globe': 21,
    'Observatory': 22, 'Nacho Tong': 23, 'Recyclomancy': 24, 'Tarot Tycoon': 25, 'Planet Tycoon': 26,
    'Money Tree': 27, 'Antimatter': 28, 'Illusion': 29, 'Petroglyph': 30, 'Retcon': 31, 'Palette': 32
}

# Card type constants
CARD_TYPES = {
    'HAND': 0, 'JOKER': 1, 'CONSUMABLE': 2, 'SHOP': 3, 
    'BOOSTER': 4, 'VOUCHER': 5, 'PADDING': 6
}

class BalatroFeatureEncoder(nn.Module):
    """
    Single transformer encoder for all Balatro cards with type embeddings and structured pointer network.
    
    Architecture:
    - Single transformer processes all cards (hand, jokers, shop items) together
    - Type embeddings distinguish between different card sources
    - Cross-component attention allows synergy detection
    - Outputs both pooled state representation and individual card embeddings for pointer network
    """
    
    def __init__(self, 
                 card_dim: int = 84,  # Divisible by 6 for 7 embeddings (7*12=84)
                 hidden_dim: int = 128,
                 num_transformer_layers: int = 3,
                 num_attention_heads: int = 12):  # 84 is divisible by 12
        super().__init__()
        
        self.card_dim = card_dim
        self.hidden_dim = hidden_dim
        
        # Card feature embeddings
        self.rank_embed = nn.Embedding(14, card_dim // 6)  # 0=unknown, 1-13=Ace-King
        self.suit_embed = nn.Embedding(5, card_dim // 6)   # 0=unknown, 1-4=SHDC
        self.enhancement_embed = nn.Embedding(9, card_dim // 6)  # 0=none, 1-8=enhancements
        self.seal_embed = nn.Embedding(5, card_dim // 6)  # 0=none, 1-4=seals (gold, red, blue, purple)
        self.edition_embed = nn.Embedding(5, card_dim // 6)  # 0=base, 1-4=editions
        
        # Joker embeddings
        self.joker_embed = nn.Embedding(VOCAB_SIZES['jokers'], card_dim // 6)
        
        # Card type embeddings (distinguish hand/joker/shop/etc)
        self.card_type_embed = nn.Embedding(7, card_dim // 6)  # 0-6=card sources
        
        # Note: Game state scalars are handled directly in game_encoder
        
        # Single transformer for all cards
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=card_dim,
            nhead=num_attention_heads,
            dim_feedforward=hidden_dim,
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        
        # Component-specific pooling
        self.component_pooling = nn.ModuleDict({
            'hand': nn.Linear(card_dim, hidden_dim),
            'joker': nn.Linear(card_dim, hidden_dim),
            'shop': nn.Linear(card_dim, hidden_dim),
            'consumable': nn.Linear(card_dim, hidden_dim),
            'booster': nn.Linear(card_dim, hidden_dim),
            'voucher': nn.Linear(card_dim, hidden_dim),
        })
        
        # Game state encoder
        self.game_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),  # Direct from 3 scalars to hidden_dim
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, observation: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the feature encoder.
        
        Args:
            observation: Dict containing:
                - 'cards': Tensor of shape [batch_size, max_cards, 6] with card features [rank, suit, enhancement, seal, edition, joker_id]
                - 'card_types': Tensor of shape [batch_size, max_cards] with card types
                - 'type_masks': Dict of masks for each card type
                - 'game_state': Tensor of shape [batch_size, 3] with game features
        
        Returns:
            Dict containing:
                - 'state_embed': Pooled state representation [batch_size, hidden_dim * 6 + hidden_dim]
                - 'card_embeddings': Individual card embeddings [batch_size, max_cards, card_dim]
                - 'card_types': Card type information
                - 'type_masks': Type masks for pooling
        """
        batch_size = observation['cards'].size(0)
        max_cards = observation['cards'].size(1)
        
        # Encode individual card features
        card_features = self._encode_cards(
            observation['cards'], 
            observation['card_types']
        )  # [batch_size, max_cards, card_dim]
        
        # Apply transformer for cross-component attention
        if max_cards > 0:
            attended_cards = self.transformer(card_features)  # [batch_size, max_cards, card_dim]
        else:
            attended_cards = torch.empty(batch_size, 0, self.card_dim, device=card_features.device)
        
        # Pool by component type for state representation
        pooled_features = []
        
        for component_type in ['hand', 'joker', 'shop', 'consumable', 'booster', 'voucher']:
            mask = observation['type_masks'][component_type]  # [batch_size, max_cards]
            pooled = self.pool_by_mask(attended_cards, mask, component_type)
            pooled_features.append(pooled)
        
        # Encode game state features
        game_features = self._encode_game_state(observation['game_state'])
        pooled_features.append(game_features)
        
        # Concatenate all features
        state_embed = torch.cat(pooled_features, dim=-1)  # [batch_size, hidden_dim * 6 + hidden_dim]
        
        return {
            'state_embed': state_embed,
            'card_embeddings': attended_cards,
            'card_types': observation['card_types'],
            'type_masks': observation['type_masks']
        }
    
    def _encode_cards(self, cards: torch.Tensor, card_types: torch.Tensor) -> torch.Tensor:
        """Encode individual card features into dense representations."""
        # cards: [batch_size, max_cards, 6] - [rank, suit, enhancement, seal, edition, joker_id]
        # card_types: [batch_size, max_cards] - card type indices
        
        batch_size, max_cards, _ = cards.shape
        
        # Extract features
        ranks = cards[..., 0].long()
        suits = cards[..., 1].long()
        enhancements = cards[..., 2].long()
        seals = cards[..., 3].long()
        editions = cards[..., 4].long()
        joker_ids = cards[..., 5].long()
        
        # Embed features
        rank_embed = self.rank_embed(ranks)  # [batch_size, max_cards, card_dim//6]
        suit_embed = self.suit_embed(suits)
        enhancement_embed = self.enhancement_embed(enhancements)
        seal_embed = self.seal_embed(seals)
        edition_embed = self.edition_embed(editions)
        joker_embed = self.joker_embed(joker_ids)
        type_embed = self.card_type_embed(card_types)
        
        # Concatenate all embeddings
        card_features = torch.cat([
            rank_embed, suit_embed, enhancement_embed, seal_embed, edition_embed, joker_embed, type_embed
        ], dim=-1)  # [batch_size, max_cards, card_dim]
        
        return card_features
    
    def _encode_game_state(self, game_state: torch.Tensor) -> torch.Tensor:
        """Encode global game state features."""
        # game_state: [batch_size, 3] - [round, ante, money]
        
        # Normalize the values to reasonable ranges
        normalized_state = torch.stack([
            torch.clamp(game_state[:, 0], 0, 20).float() / 20.0,  # round
            torch.clamp(game_state[:, 1], 0, 20).float() / 20.0,  # ante  
            torch.clamp(game_state[:, 2], 0, 999).float() / 999.0,  # money
        ], dim=-1)
        
        game_features = self.game_encoder(normalized_state)
        
        return game_features
    
    def pool_by_mask(self, 
                    card_embeddings: torch.Tensor, 
                    mask: torch.Tensor, 
                    component_type: str) -> torch.Tensor:
        """
        Pool card embeddings by component type using attention-based pooling.
        
        Args:
            card_embeddings: [batch_size, max_cards, card_dim]
            mask: [batch_size, max_cards] - boolean mask for valid cards
            component_type: string identifier for component type
            
        Returns:
            Pooled representation: [batch_size, hidden_dim]
        """
        batch_size = card_embeddings.size(0)
        
        # Handle empty case
        if mask.sum() == 0:
            return torch.zeros(batch_size, self.hidden_dim, device=card_embeddings.device)
        
        # Mask out invalid cards
        masked_embeddings = card_embeddings * mask.unsqueeze(-1).float()
        
        # Compute attention weights
        attention_weights = F.softmax(
            (masked_embeddings.sum(dim=-1) + (~mask).float() * -1e9), 
            dim=-1
        )  # [batch_size, max_cards]
        
        # Weighted sum
        pooled = torch.sum(
            masked_embeddings * attention_weights.unsqueeze(-1), 
            dim=1
        )  # [batch_size, card_dim]
        
        # Project to hidden dimension
        pooled = self.component_pooling[component_type](pooled)  # [batch_size, hidden_dim]
        
        return pooled


def extract_edition(item: Dict) -> int:
    """Extract edition from item's edition dict."""
    edition_dict = item.get('edition', {})
    if edition_dict.get('foil'): return 1
    elif edition_dict.get('holo'): return 2  
    elif edition_dict.get('polychrome'): return 3
    elif edition_dict.get('negative'): return 4
    return 0  # base

def extract_card_features(card: Dict) -> List[int]:
    """Extract features from a playing card: [rank, suit, enhancement, seal, edition, joker_id]"""
    if 'value' not in card or 'suit' not in card:
        raise KeyError(f"Missing 'value' or 'suit' in card: {card}")
    
    rank = RANK_VOCAB.get(card['value'])
    suit = SUIT_VOCAB.get(card['suit'])
    if rank is None: raise ValueError(f"Unknown card value: {card['value']}")
    if suit is None: raise ValueError(f"Unknown card suit: {card['suit']}")
    
    enhancement = ENHANCEMENT_VOCAB.get(card.get('ability_name', 'Default Base'))
    if enhancement is None: raise ValueError(f"Unknown enhancement: {card.get('ability_name')}")
    
    seal = SEAL_VOCAB.get(card.get('seal', 'none'))
    if seal is None: raise ValueError(f"Unknown seal: {card.get('seal')}")
    
    edition = extract_edition(card)
    return [rank, suit, enhancement, seal, edition, 0]  # No joker ID

def extract_joker_features(joker: Dict) -> List[int]:
    """Extract features from a joker: [0, 0, 0, 0, edition, joker_id]"""
    if 'name' not in joker:
        raise KeyError(f"'name' missing from joker: {joker}")
    
    joker_id = JOKER_VOCAB.get(joker['name'])
    if joker_id is None:
        raise ValueError(f"Unknown joker: {joker['name']}")
    
    edition = extract_edition(joker)
    return [0, 0, 0, 0, edition, joker_id]

def extract_item_features(item: Dict) -> Tuple[List[int], int]:
    """Extract features from any item and return (features, card_type)."""
    # Detect item type and extract features
    if 'value' in item and 'suit' in item:
        return extract_card_features(item), CARD_TYPES['HAND']
    elif 'name' in item:
        if item['name'] in JOKER_VOCAB:
            return extract_joker_features(item), CARD_TYPES['JOKER']
        elif item['name'] in CONSUMABLE_VOCAB:
            consumable_id = CONSUMABLE_VOCAB[item['name']]
            return [0, 0, 0, 0, 0, consumable_id], CARD_TYPES['CONSUMABLE']
        elif item['name'] in BOOSTER_VOCAB:
            booster_id = BOOSTER_VOCAB[item['name']]
            return [0, 0, booster_id, 0, 0, 0], CARD_TYPES['BOOSTER']
        elif item['name'] in VOUCHER_VOCAB:
            voucher_id = VOUCHER_VOCAB[item['name']]
            return [0, 0, voucher_id, 0, 0, 0], CARD_TYPES['VOUCHER']
        else:
            raise ValueError(f"Unknown item: {item['name']}")
    else:
        raise ValueError(f"Cannot identify item type: {item}")

class CardFeatureExtractor:
    """Simplified utility class to extract structured features from Balatro game state."""
    
    @staticmethod
    def extract_features(game_state: Dict) -> Dict[str, any]:
        """Extract structured features from game state for the feature encoder."""
        # Validate game state
        if 'game' not in game_state:
            raise KeyError("'game' key missing from game_state")
        game = game_state['game']
        for key in ['round', 'ante', 'dollars']:
            if key not in game:
                raise KeyError(f"'{key}' key missing from game state")
        
        features = {
            'cards': [],
            'card_types': [],
            'type_masks': {k: [] for k in ['hand', 'joker', 'shop', 'consumable', 'booster', 'voucher']},
            'game_state': [game['round'], game['ante'], game['dollars']]
        }
        
        # Helper to add item with type masks
        def add_item(item, source_type):
            item_features, card_type = extract_item_features(item)
            features['cards'].append(item_features)
            features['card_types'].append(card_type)
            
            # Set all masks to 0 initially
            mask_values = {k: 0 for k in features['type_masks']}
            
            # Set masks based on card type and source
            type_name = [k for k, v in CARD_TYPES.items() if v == card_type][0].lower()
            mask_values[type_name] = 1
            if source_type == 'shop':
                mask_values['shop'] = 1
            
            # Append mask values
            for mask_type, value in mask_values.items():
                features['type_masks'][mask_type].append(value)
        
        # Extract from all sources
        for source, items in [
            ('hand', game_state.get('hand', [])),
            ('joker', game_state.get('jokers', [])),
            ('consumable', game_state.get('consumables', [])),
        ]:
            for item in items:
                add_item(item, source)
        
        # Shop items (mixed types)
        shop = game_state.get('shop', {})
        for item in shop.get('jokers', []) + shop.get('boosters', []) + shop.get('vouchers', []):
            add_item(item, 'shop')
        
        return features
