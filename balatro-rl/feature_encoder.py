import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np

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
                 card_dim: int = 64,
                 hidden_dim: int = 128,
                 num_transformer_layers: int = 3,
                 num_attention_heads: int = 8):
        super().__init__()
        
        self.card_dim = card_dim
        self.hidden_dim = hidden_dim
        
        # Card feature embeddings
        self.rank_embed = nn.Embedding(14, card_dim // 4)  # 0=unknown, 1-13=Ace-King
        self.suit_embed = nn.Embedding(5, card_dim // 4)   # 0=unknown, 1-4=SHDC
        self.enhancement_embed = nn.Embedding(21, card_dim // 4)  # 0=none, 1-20=enhancements
        self.seal_embed = nn.Embedding(7, card_dim // 4)  # 0=none, 1-6=seal types
        
        # Card type embeddings (distinguish hand/joker/shop/etc)
        self.card_type_embed = nn.Embedding(7, card_dim // 4)  # 0-6=card sources
        
        # Game state embeddings
        self.round_embed = nn.Embedding(20, card_dim // 4)  # Round 1-20
        self.ante_embed = nn.Embedding(20, card_dim // 4)    # Ante 1-20
        self.money_embed = nn.Embedding(100, card_dim // 4)  # Money 0-99
        
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
            nn.Linear(card_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, observation: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the feature encoder.
        
        Args:
            observation: Dict containing:
                - 'cards': Tensor of shape [batch_size, max_cards, 4] with card features
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
            # Create attention mask for valid cards
            valid_mask = (observation['cards'].sum(dim=-1) != 0)  # [batch_size, max_cards]
            attended_cards = self.transformer(
                card_features,
                src_key_padding_mask=~valid_mask
            )  # [batch_size, max_cards, card_dim]
        else:
            attended_cards = card_features
        
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
        # cards: [batch_size, max_cards, 4] - [rank, suit, enhancement, seal]
        # card_types: [batch_size, max_cards] - card type indices
        
        batch_size, max_cards, _ = cards.shape
        
        # Extract features
        ranks = cards[..., 0].long()
        suits = cards[..., 1].long()
        enhancements = cards[..., 2].long()
        seals = cards[..., 3].long()
        
        # Embed features
        rank_embed = self.rank_embed(ranks)  # [batch_size, max_cards, card_dim//4]
        suit_embed = self.suit_embed(suits)
        enhancement_embed = self.enhancement_embed(enhancements)
        seal_embed = self.seal_embed(seals)
        type_embed = self.card_type_embed(card_types)
        
        # Concatenate all embeddings
        card_features = torch.cat([
            rank_embed, suit_embed, enhancement_embed, seal_embed, type_embed
        ], dim=-1)  # [batch_size, max_cards, card_dim]
        
        return card_features
    
    def _encode_game_state(self, game_state: torch.Tensor) -> torch.Tensor:
        """Encode global game state features."""
        # game_state: [batch_size, 3] - [round, ante, money]
        
        rounds = game_state[:, 0].long()
        antes = game_state[:, 1].long()
        money = torch.clamp(game_state[:, 2].long(), 0, 99)
        
        round_embed = self.round_embed(rounds)
        ante_embed = self.ante_embed(antes)
        money_embed = self.money_embed(money)
        
        game_features = torch.cat([round_embed, ante_embed, money_embed], dim=-1)
        game_features = self.game_encoder(game_features)
        
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


class CardFeatureExtractor:
    """
    Utility class to extract structured features from Balatro game state.
    """
    
    @staticmethod
    def extract_features(game_state: Dict) -> Dict[str, any]:
        """
        Extract structured features from game state for the feature encoder.
        
        Args:
            game_state: Raw game state dictionary from controller
            
        Returns:
            Dict with structured features ready for tensor conversion
        """
        features = {
            'cards': [],
            'card_types': [],
            'type_masks': {
                'hand': [],
                'joker': [],
                'shop': [],
                'consumable': [],
                'booster': [],
                'voucher': []
            },
            'game_state': []
        }
        
        # Extract game state features
        game = game_state.get('game', {})
        round_num = game.get('round', 1)
        ante = game.get('ante', 1)
        money = game_state.get('dollars', 0)
        
        features['game_state'] = [round_num, ante, money]
        
        # Track card index across all sources
        card_idx = 0
        
        # Extract hand cards
        hand_cards = game_state.get('hand', [])
        for card in hand_cards:
            card_features = CardFeatureExtractor._extract_card_features(card)
            features['cards'].append(card_features)
            features['card_types'].append(0)  # HAND
            features['type_masks']['hand'].append(1)
            features['type_masks']['joker'].append(0)
            features['type_masks']['shop'].append(0)
            features['type_masks']['consumable'].append(0)
            features['type_masks']['booster'].append(0)
            features['type_masks']['voucher'].append(0)
            card_idx += 1
        
        # Extract jokers
        jokers = game_state.get('jokers', [])
        for joker in jokers:
            card_features = CardFeatureExtractor._extract_joker_features(joker)
            features['cards'].append(card_features)
            features['card_types'].append(1)  # JOKER
            features['type_masks']['hand'].append(0)
            features['type_masks']['joker'].append(1)
            features['type_masks']['shop'].append(0)
            features['type_masks']['consumable'].append(0)
            features['type_masks']['booster'].append(0)
            features['type_masks']['voucher'].append(0)
            card_idx += 1
        
        # Extract shop items
        shop = game_state.get('shop', {})
        shop_cards = shop.get('jokers', [])
        for card in shop_cards:
            card_features = CardFeatureExtractor._extract_shop_features(card)
            features['cards'].append(card_features)
            features['card_types'].append(3)  # SHOP
            features['type_masks']['hand'].append(0)
            features['type_masks']['joker'].append(0)
            features['type_masks']['shop'].append(1)
            features['type_masks']['consumable'].append(0)
            features['type_masks']['booster'].append(0)
            features['type_masks']['voucher'].append(0)
            card_idx += 1
        
        # Extract consumables
        consumables = game_state.get('consumables', [])
        for consumable in consumables:
            card_features = CardFeatureExtractor._extract_consumable_features(consumable)
            features['cards'].append(card_features)
            features['card_types'].append(2)  # CONSUMABLE
            features['type_masks']['hand'].append(0)
            features['type_masks']['joker'].append(0)
            features['type_masks']['shop'].append(0)
            features['type_masks']['consumable'].append(1)
            features['type_masks']['booster'].append(0)
            features['type_masks']['voucher'].append(0)
            card_idx += 1
        
        # Extract boosters
        boosters = shop.get('boosters', [])
        for booster in boosters:
            card_features = CardFeatureExtractor._extract_booster_features(booster)
            features['cards'].append(card_features)
            features['card_types'].append(4)  # BOOSTER
            features['type_masks']['hand'].append(0)
            features['type_masks']['joker'].append(0)
            features['type_masks']['shop'].append(0)
            features['type_masks']['consumable'].append(0)
            features['type_masks']['booster'].append(1)
            features['type_masks']['voucher'].append(0)
            card_idx += 1
        
        # Extract vouchers
        vouchers = shop.get('vouchers', [])
        for voucher in vouchers:
            card_features = CardFeatureExtractor._extract_voucher_features(voucher)
            features['cards'].append(card_features)
            features['card_types'].append(5)  # VOUCHER
            features['type_masks']['hand'].append(0)
            features['type_masks']['joker'].append(0)
            features['type_masks']['shop'].append(0)
            features['type_masks']['consumable'].append(0)
            features['type_masks']['booster'].append(0)
            features['type_masks']['voucher'].append(1)
            card_idx += 1
        
        # Pad to consistent length
        max_cards = 50  # Conservative upper bound
        while len(features['cards']) < max_cards:
            features['cards'].append([0, 0, 0, 0])  # Padding
            features['card_types'].append(6)  # PADDING type
            for mask_type in features['type_masks']:
                features['type_masks'][mask_type].append(0)
        
        return features
    
    @staticmethod
    def _extract_card_features(card: Dict) -> List[int]:
        """Extract features from a playing card."""
        # Map card values to ranks
        value_map = {
            'Ace': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
            '8': 8, '9': 9, '10': 10, 'Jack': 11, 'Queen': 12, 'King': 13
        }
        
        rank = value_map.get(card.get('value', ''), 0)
        
        # Map suits
        suit_map = {'Spades': 1, 'Hearts': 2, 'Diamonds': 3, 'Clubs': 4}
        suit = suit_map.get(card.get('suit', ''), 0)
        
        # Map enhancements
        enhancement_map = {
            'Default Base': 0, 'Bonus Card': 1, 'Mult Card': 2, 'Wild Card': 3,
            'Glass Card': 4, 'Steel Card': 5, 'Stone Card': 6, 'Gold Card': 7,
            'Lucky Card': 8, 'Foil': 9, 'Holographic': 10, 'Polychrome': 11,
            'Negative': 12, 'Eternal': 13, 'Perishable': 14, 'Rental': 15
        }
        enhancement = enhancement_map.get(card.get('ability_name', 'Default Base'), 0)
        
        # Map seals
        seal_map = {'none': 0, 'Red': 1, 'Blue': 2, 'Gold': 3, 'Purple': 4, 'Green': 5}
        seal = seal_map.get(card.get('seal', 'none'), 0)
        
        return [rank, suit, enhancement, seal]
    
    @staticmethod
    def _extract_joker_features(joker: Dict) -> List[int]:
        """Extract features from a joker card."""
        # Jokers don't have traditional card features
        return [0, 0, 0, 0]  # Placeholder
    
    @staticmethod
    def _extract_shop_features(card: Dict) -> List[int]:
        """Extract features from shop cards."""
        # Similar to regular cards but may include jokers
        return CardFeatureExtractor._extract_card_features(card)
    
    @staticmethod
    def _extract_consumable_features(consumable: Dict) -> List[int]:
        """Extract features from consumable cards."""
        # Consumables don't have traditional card features
        return [0, 0, 0, 0]  # Placeholder
    
    @staticmethod
    def _extract_booster_features(booster: Dict) -> List[int]:
        """Extract features from booster packs."""
        # Boosters don't have traditional card features
        return [0, 0, 0, 0]  # Placeholder
    
    @staticmethod
    def _extract_voucher_features(voucher: Dict) -> List[int]:
        """Extract features from vouchers."""
        # Vouchers don't have traditional card features
        return [0, 0, 0, 0]  # Placeholder
