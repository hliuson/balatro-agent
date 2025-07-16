"""
Text Encoder Module for Balatro TorchRL Agent.

This module provides a TensorDictModule wrapper around the text encoding functionality
extracted from the BalatroPolicy. It handles tokenization, LLM encoding, and projection
to the desired embedding dimension.
"""

import torch
import torch.nn as nn
from typing import Optional
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from transformers import AutoTokenizer, AutoModelForCausalLM

# Model architecture constants
ENCODER_DIM = 512


class TextEncoder(nn.Module):
    """
    Text encoder that converts text observations to embeddings.
    
    This module handles:
    1. Tokenization of text input
    2. LLM encoding to get text embeddings  
    3. Projection to fixed ENCODER_DIM size
    """
    
    def __init__(self, llm_model: str = "Qwen/Qwen3-0.6B", device: str = "auto", freeze_encoder: bool = True):
        """
        Initialize the text encoder.
        
        Args:
            llm_model: HuggingFace model name for text encoding
            device: Device to place the model on
            freeze_encoder: Whether to freeze the LLM encoder weights
        """
        super().__init__()
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Initialize tokenizer and LLM encoder
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model)
        self.encoder = AutoModelForCausalLM.from_pretrained(llm_model).to(self.device)
        
        # Add pad token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Freeze encoder if requested
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Projection layer to map encoder output to ENCODER_DIM
        encoder_hidden_size = self.encoder.config.hidden_size
        self.projection = nn.Linear(encoder_hidden_size, ENCODER_DIM)
        
        # Move to device
        self.to(self.device)
    
    def forward(self, observation) -> torch.Tensor:
        """
        Encode text observation to embedding.
        
        Args:
            observation: Text string from environment (may be wrapped in list)
            
        Returns:
            Embedding tensor of shape [ENCODER_DIM]
        """
        # Handle observation format (may be wrapped in list from NonTensor)
        if isinstance(observation, list):
            observation = observation[0]
        elif hasattr(observation, 'item'):
            observation = observation.item()
        
        # Ensure it's a string
        if not isinstance(observation, str):
            observation = str(observation)
        
        # Tokenize input text
        with torch.no_grad():
            tokens = self.tokenizer(
                observation, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            ).to(self.device)
            
            # Get LLM encoding
            encoder_outputs = self.encoder(**tokens, output_hidden_states=True)
            
            # Use last token's hidden state (last position in sequence)
            text_embedding = encoder_outputs.hidden_states[-1][:, -1, :]  #hidden_states is a tuple over layers. 

        # Project to ENCODER_DIM
        projected_embedding = self.projection(text_embedding)  # [1, ENCODER_DIM]
        
        return projected_embedding.squeeze(0)  # [ENCODER_DIM]


def make_text_encoder_module(
    llm_model: str = "Qwen/Qwen3-0.6B",
    device: str = "auto", 
    freeze_encoder: bool = True
) -> TensorDictModule:
    """
    Factory function to create a TensorDictModule text encoder.
    
    Args:
        llm_model: HuggingFace model name for text encoding
        device: Device to place the model on
        freeze_encoder: Whether to freeze the LLM encoder weights
        
    Returns:
        TensorDictModule that maps observation -> embeddings
    """
    # Create the encoder network
    encoder_net = TextEncoder(
        llm_model=llm_model,
        device=device,
        freeze_encoder=freeze_encoder
    )
    
    # Wrap in TensorDictModule
    text_encoder_module = TensorDictModule(
        module=encoder_net,
        in_keys=["observation"],
        out_keys=["embeddings"]
    )
    
    return text_encoder_module


class CardEncoder(nn.Module):
    """
    Card encoder that converts card dictionaries to embeddings.
    
    This is used for card selection tasks where we need to embed
    individual cards for combinatorial selection.
    """
    
    def __init__(self, text_encoder: TextEncoder):
        """
        Initialize card encoder with shared text encoder.
        
        Args:
            text_encoder: Shared TextEncoder instance
        """
        super().__init__()
        self.text_encoder = text_encoder
        self.device = text_encoder.device
    
    def forward(self, cards: list) -> torch.Tensor:
        """
        Embed a list of card dictionaries.
        
        Args:
            cards: List of card dictionaries from game state
            
        Returns:
            Card embeddings tensor of shape [num_cards, ENCODER_DIM]
        """
        if cards is None or len(cards) == 0:
            return torch.empty(0, ENCODER_DIM, device=self.device)
        
        # Format cards as text descriptions
        card_texts = []
        for card in cards:
            if card.get('name'):
                # Use card name if available
                card_text = card['name']
            else:
                # Build description from card properties
                value = card.get('value', '')
                suit = card.get('suit', '')
                if value and suit:
                    card_text = f"{value} of {suit}"
                else:
                    card_text = str(card)
            
            # Add additional card properties
            enhancement = card.get('ability_name', '')
            if enhancement and enhancement != "Default Base":
                card_text += f" ({enhancement})"
            
            seal = card.get('seal', 'none')
            if seal and seal != "none":
                card_text += f" [{seal} Seal]"
            
            card_texts.append(card_text)
        
        # Encode all cards using the text encoder
        card_embeddings = []
        for card_text in card_texts:
            embedding = self.text_encoder(card_text)
            card_embeddings.append(embedding.unsqueeze(0))
        
        return torch.cat(card_embeddings, dim=0)  # [num_cards, ENCODER_DIM]


def make_card_encoder_module(text_encoder: TextEncoder) -> TensorDictModule:
    """
    Factory function to create a TensorDictModule card encoder.
    
    Args:
        text_encoder: Shared TextEncoder instance
        
    Returns:
        TensorDictModule that maps cards -> card_embeddings
    """
    # Create the card encoder network
    card_encoder_net = CardEncoder(text_encoder)
    
    # Wrap in TensorDictModule
    card_encoder_module = TensorDictModule(
        module=card_encoder_net,
        in_keys=["cards"],
        out_keys=["card_embeddings"]
    )
    
    return card_encoder_module


if __name__ == "__main__":
    # Test the text encoder module
    print("Testing TextEncoder...")
    
    # Create text encoder
    text_encoder_module = make_text_encoder_module(device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Test input
    test_observation = "Game State: Ante 1, Round 1. Hand: 2 of Hearts, 3 of Spades. Chips: 100."
    
    # Create TensorDict input
    input_td = TensorDict({
        "observation": test_observation
    }, batch_size=[])
    
    # Test forward pass
    print("Input observation:", test_observation[:50] + "...")
    
    output_td = text_encoder_module(input_td)
    print("Output keys:", list(output_td.keys()))
    print("Embeddings shape:", output_td["embeddings"].shape)
    print("Embeddings dtype:", output_td["embeddings"].dtype)
    print("Embeddings device:", output_td["embeddings"].device)
    
    # Test card encoder
    print("\nTesting CardEncoder...")
    
    # Get the underlying text encoder
    text_encoder = text_encoder_module.module
    card_encoder_module = make_card_encoder_module(text_encoder)
    
    # Test cards
    test_cards = [
        {"name": "Ace of Spades", "ability_name": "Steel Card", "seal": "gold"},
        {"value": "2", "suit": "Hearts", "ability_name": "Default Base", "seal": "none"},
        {"name": "Joker", "ability_name": "Hack"}
    ]
    
    # Create TensorDict input for cards
    card_input_td = TensorDict({
        "cards": test_cards
    }, batch_size=[])
    
    # Test card encoding
    card_output_td = card_encoder_module(card_input_td)
    print("Card embeddings shape:", card_output_td["card_embeddings"].shape)
    print("Card embeddings dtype:", card_output_td["card_embeddings"].dtype)
    
    print("TextEncoder testing complete!")