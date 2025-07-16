"""
Action Head Module for Balatro TorchRL Agent.

This module provides the action selection head that converts LSTM embeddings
to action logits for the 21 possible Balatro actions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Any
from tensordict import TensorDict
from tensordict.nn import TensorDictModule

# Model architecture constants
HIDDEN_DIM = 256
ACTION_DIM = 21  # Number of discrete action types (from Actions enum)
ENCODER_DIM = 512  # For card embeddings
COMBINATORIAL_DIM = 128  # Dimension for combinatorial selection
NUM_LAYERS = 2  # For combinatorial LSTM


class ActionHead(nn.Module):
    """
    Action head that converts LSTM embeddings to action logits.
    
    Takes LSTM hidden state embeddings and produces logits for the 21 possible actions.
    """
    
    def __init__(
        self, 
        input_size: int = HIDDEN_DIM, 
        action_dim: int = ACTION_DIM,
        device: str = "auto"
    ):
        """
        Initialize the action head.
        
        Args:
            input_size: Size of input embeddings from LSTM
            action_dim: Number of possible actions (21 for Balatro)
            device: Device to place the module on
        """
        super().__init__()
        
        # Set device
        if device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)
        
        # Simple linear projection to action logits
        self.action_projection = nn.Linear(input_size, action_dim)
        
        # Move to device
        self.to(self._device)
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Convert embeddings to action logits.
        
        Args:
            embeddings: LSTM output embeddings of shape [..., input_size]
            
        Returns:
            Action logits tensor of shape [..., action_dim]
        """
        return self.action_projection(embeddings)


class CombinatorialCardSelector(nn.Module):
    """
    Combinatorial card selector using autoregressive generation.
    
    This module handles card selection for actions that require choosing
    specific cards from available options (hand, jokers, shop items, etc.).
    """
    
    def __init__(
        self,
        context_size: int = HIDDEN_DIM,
        card_embedding_size: int = ENCODER_DIM,
        hidden_size: int = COMBINATORIAL_DIM,
        num_layers: int = NUM_LAYERS,
        device: str = "auto"
    ):
        """
        Initialize the combinatorial card selector.
        
        Args:
            context_size: Size of context embeddings from main LSTM
            card_embedding_size: Size of card embeddings
            hidden_size: Size of combinatorial LSTM hidden state
            num_layers: Number of LSTM layers
            device: Device to place the module on
        """
        super().__init__()
        
        # Set device
        if device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)
        
        # Project context to card embedding space for initial input
        self.context_projection = nn.Linear(context_size, card_embedding_size)
        
        # Combinatorial LSTM for autoregressive selection
        self.combinatorial_lstm = nn.LSTM(
            input_size=card_embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Project LSTM output back to card embedding space for similarity computation
        self.output_projection = nn.Linear(hidden_size, card_embedding_size)
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Move to device
        self.to(self._device)
    
    def forward(
        self, 
        context_embeddings: torch.Tensor,
        card_embeddings: torch.Tensor,
        max_selections: int = 5,
        temperature: float = 1.0
    ) -> List[int]:
        """
        Select cards using autoregressive generation.
        
        Args:
            context_embeddings: Context from main LSTM [batch_size, context_size]
            card_embeddings: Available card embeddings [num_cards, card_embedding_size]
            max_selections: Maximum number of cards to select
            temperature: Temperature for sampling (lower = more deterministic)
            
        Returns:
            List of selected card indices
        """
        if card_embeddings.size(0) == 0:
            return []
        
        batch_size = context_embeddings.size(0) if context_embeddings.dim() > 1 else 1
        
        # Handle single sample case
        if context_embeddings.dim() == 1:
            context_embeddings = context_embeddings.unsqueeze(0)
        
        # Project context to card embedding space
        initial_input = self.context_projection(context_embeddings)  # [batch_size, card_embedding_size]
        initial_input = initial_input.unsqueeze(1)  # [batch_size, 1, card_embedding_size]
        
        # Initialize LSTM hidden state
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self._device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self._device)
        
        # Get initial LSTM output from context
        lstm_out, (h_n, c_n) = self.combinatorial_lstm(initial_input, (h_0, c_0))
        
        selected_indices = []
        available_mask = torch.ones(card_embeddings.size(0), dtype=torch.bool, device=self._device)
        
        for step in range(min(max_selections, card_embeddings.size(0))):
            # Project LSTM output to card embedding space
            projected_output = self.output_projection(lstm_out[:, -1, :])  # [batch_size, card_embedding_size]
            
            # Compute similarities with available cards
            similarities = torch.matmul(projected_output, card_embeddings.T)  # [batch_size, num_cards]
            
            # Apply temperature scaling
            logits = similarities / temperature
            
            # Mask already selected cards
            logits = logits.masked_fill(~available_mask.unsqueeze(0), float('-inf'))
            
            # Check if any cards are available
            if available_mask.sum() == 0:
                break
            
            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            
            # Handle single batch case
            if batch_size == 1:
                probs = probs.squeeze(0)
            
            selected_idx = torch.multinomial(probs, num_samples=1).item()
            selected_indices.append(selected_idx)
            
            # Update availability mask
            available_mask[selected_idx] = False
            
            # Prepare next LSTM input (selected card embedding)
            selected_card_embedding = card_embeddings[selected_idx].unsqueeze(0).unsqueeze(0)  # [1, 1, card_embedding_size]
            
            # Expand to batch size if needed
            if batch_size > 1:
                selected_card_embedding = selected_card_embedding.expand(batch_size, 1, -1)
            
            # Continue LSTM
            lstm_out, (h_n, c_n) = self.combinatorial_lstm(selected_card_embedding, (h_n, c_n))
        
        return selected_indices
    
    def forward_batch(
        self,
        context_embeddings: torch.Tensor,
        card_embeddings_list: List[torch.Tensor],
        max_selections: int = 5,
        temperature: float = 1.0
    ) -> List[List[int]]:
        """
        Batch version of forward for processing multiple card selection scenarios.
        
        Args:
            context_embeddings: Context from main LSTM [batch_size, context_size]
            card_embeddings_list: List of card embeddings for each batch item
            max_selections: Maximum number of cards to select
            temperature: Temperature for sampling
            
        Returns:
            List of selected card indices for each batch item
        """
        batch_size = context_embeddings.size(0)
        results = []
        
        for i in range(batch_size):
            context = context_embeddings[i]
            card_embeddings = card_embeddings_list[i] if i < len(card_embeddings_list) else torch.empty(0, ENCODER_DIM, device=self._device)
            
            selected = self.forward(
                context_embeddings=context,
                card_embeddings=card_embeddings,
                max_selections=max_selections,
                temperature=temperature
            )
            results.append(selected)
        
        return results


class ValueHead(nn.Module):
    """
    Value head for critic network in PPO.
    
    Takes LSTM hidden state embeddings and produces state value estimates.
    """
    
    def __init__(
        self, 
        input_size: int = HIDDEN_DIM,
        device: str = "auto"
    ):
        """
        Initialize the value head.
        
        Args:
            input_size: Size of input embeddings from LSTM
            device: Device to place the module on
        """
        super().__init__()
        
        # Set device
        if device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)
        
        # Value projection with small hidden layer
        self.value_net = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.ReLU(),
            nn.Linear(input_size // 2, 1)
        )
        
        # Move to device
        self.to(self._device)
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Convert embeddings to value estimate.
        
        Args:
            embeddings: LSTM output embeddings of shape [..., input_size]
            
        Returns:
            Value estimates tensor of shape [..., 1]
        """
        return self.value_net(embeddings)


def make_action_head_module(
    input_size: int = HIDDEN_DIM,
    action_dim: int = ACTION_DIM,
    device: str = "auto"
) -> TensorDictModule:
    """
    Factory function to create a TensorDictModule action head.
    
    Args:
        input_size: Size of input embeddings from LSTM
        action_dim: Number of possible actions 
        device: Device to place the module on
        
    Returns:
        TensorDictModule that maps lstm_embeddings -> logits
    """
    # Create the action head network
    action_head_net = ActionHead(
        input_size=input_size,
        action_dim=action_dim,
        device=device
    )
    
    # Wrap in TensorDictModule
    action_head_module = TensorDictModule(
        module=action_head_net,
        in_keys=["lstm_embeddings"],
        out_keys=["logits"]
    )
    
    return action_head_module


def make_value_head_module(
    input_size: int = HIDDEN_DIM,
    device: str = "auto"
) -> TensorDictModule:
    """
    Factory function to create a TensorDictModule value head.
    
    Args:
        input_size: Size of input embeddings from LSTM
        device: Device to place the module on
        
    Returns:
        TensorDictModule that maps lstm_embeddings -> state_value
    """
    # Create the value head network
    value_head_net = ValueHead(
        input_size=input_size,
        device=device
    )
    
    # Wrap in TensorDictModule
    value_head_module = TensorDictModule(
        module=value_head_net,
        in_keys=["lstm_embeddings"],
        out_keys=["state_value"]
    )
    
    return value_head_module


def make_combinatorial_card_selector(
    context_size: int = HIDDEN_DIM,
    card_embedding_size: int = ENCODER_DIM,
    hidden_size: int = COMBINATORIAL_DIM,
    num_layers: int = NUM_LAYERS,
    device: str = "auto"
) -> CombinatorialCardSelector:
    """
    Factory function to create a combinatorial card selector.
    
    Args:
        context_size: Size of context embeddings from main LSTM
        card_embedding_size: Size of card embeddings
        hidden_size: Size of combinatorial LSTM hidden state
        num_layers: Number of LSTM layers
        device: Device to place the module on
        
    Returns:
        CombinatorialCardSelector for card selection
    """
    return CombinatorialCardSelector(
        context_size=context_size,
        card_embedding_size=card_embedding_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        device=device
    )


if __name__ == "__main__":
    # Test the action head modules
    print("Testing Action Head modules...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Test Action Head
    print("\n=== Testing Action Head ===")
    action_head_module = make_action_head_module(device=device)
    
    # Test input (LSTM embeddings)
    test_embeddings = torch.randn(HIDDEN_DIM, device=device)
    
    # Create TensorDict input
    input_td = TensorDict({
        "lstm_embeddings": test_embeddings
    }, batch_size=[], device=device)
    
    print("Input embeddings shape:", input_td["lstm_embeddings"].shape)
    
    # Forward pass
    output_td = action_head_module(input_td)
    print("Output keys:", list(output_td.keys()))
    print("Action logits shape:", output_td["logits"].shape)
    print("Action logits dtype:", output_td["logits"].dtype)
    print("Action logits device:", output_td["logits"].device)
    
    # Test with batch dimension
    print("\n=== Testing Batch Processing ===")
    batch_embeddings = torch.randn(4, HIDDEN_DIM, device=device)  # Batch of 4
    
    batch_input_td = TensorDict({
        "lstm_embeddings": batch_embeddings
    }, batch_size=[4], device=device)
    
    print("Batch input embeddings shape:", batch_input_td["lstm_embeddings"].shape)
    batch_output_td = action_head_module(batch_input_td)
    print("Batch output logits shape:", batch_output_td["logits"].shape)
    
    # Test Value Head
    print("\n=== Testing Value Head ===")
    value_head_module = make_value_head_module(device=device)
    
    # Test single value
    value_output_td = value_head_module(input_td)
    print("Value output keys:", list(value_output_td.keys()))
    print("State value shape:", value_output_td["state_value"].shape)
    print("State value dtype:", value_output_td["state_value"].dtype)
    
    # Test batch values
    batch_value_output_td = value_head_module(batch_input_td)
    print("Batch state values shape:", batch_value_output_td["state_value"].shape)
    
    # Test action probability distribution
    print("\n=== Testing Action Distribution ===")
    logits = output_td["logits"]
    action_probs = torch.softmax(logits, dim=-1)
    print("Action probabilities shape:", action_probs.shape)
    print("Action probabilities sum:", action_probs.sum().item())
    print("Top 3 actions (indices):", torch.topk(action_probs, 3).indices.tolist())
    print("Top 3 probabilities:", torch.topk(action_probs, 3).values.tolist())
    
    # Test action sampling
    print("\n=== Testing Action Sampling ===")
    sampled_action = torch.multinomial(action_probs, num_samples=1)
    print("Sampled action index:", sampled_action.item())
    print("Sampled action probability:", action_probs[sampled_action].item())
    
    # Test with sequential inputs (time dimension)
    print("\n=== Testing Sequential Processing ===")
    seq_embeddings = torch.randn(1, 5, HIDDEN_DIM, device=device)  # [batch=1, seq=5, features]
    
    seq_input_td = TensorDict({
        "lstm_embeddings": seq_embeddings
    }, batch_size=[1], device=device)
    
    print("Sequential input shape:", seq_input_td["lstm_embeddings"].shape)
    seq_output_td = action_head_module(seq_input_td)
    print("Sequential output logits shape:", seq_output_td["logits"].shape)
    
    seq_value_output_td = value_head_module(seq_input_td)
    print("Sequential state values shape:", seq_value_output_td["state_value"].shape)
    
    # Test Combinatorial Card Selector
    print("\n=== Testing Combinatorial Card Selector ===")
    card_selector = make_combinatorial_card_selector(device=device)
    
    # Create test card embeddings (simulating 5 cards)
    test_card_embeddings = torch.randn(5, ENCODER_DIM, device=device)
    print("Test card embeddings shape:", test_card_embeddings.shape)
    
    # Test single selection
    context = torch.randn(HIDDEN_DIM, device=device)
    selected_cards = card_selector.forward(
        context_embeddings=context,
        card_embeddings=test_card_embeddings,
        max_selections=3,
        temperature=1.0
    )
    print("Selected card indices:", selected_cards)
    print("Number of selected cards:", len(selected_cards))
    
    # Test batch selection
    print("\n=== Testing Batch Card Selection ===")
    batch_context = torch.randn(2, HIDDEN_DIM, device=device)
    batch_card_embeddings = [
        torch.randn(3, ENCODER_DIM, device=device),  # 3 cards for first batch item
        torch.randn(4, ENCODER_DIM, device=device),  # 4 cards for second batch item
    ]
    
    batch_selected = card_selector.forward_batch(
        context_embeddings=batch_context,
        card_embeddings_list=batch_card_embeddings,
        max_selections=2,
        temperature=0.8
    )
    print("Batch selected cards:", batch_selected)
    print("Batch selection lengths:", [len(selection) for selection in batch_selected])
    
    # Test with empty card embeddings
    print("\n=== Testing Empty Card Selection ===")
    empty_cards = torch.empty(0, ENCODER_DIM, device=device)
    empty_selection = card_selector.forward(
        context_embeddings=context,
        card_embeddings=empty_cards,
        max_selections=3
    )
    print("Empty card selection:", empty_selection)
    
    # Test different temperatures
    print("\n=== Testing Temperature Effects ===")
    for temp in [0.1, 1.0, 2.0]:
        temp_selection = card_selector.forward(
            context_embeddings=context,
            card_embeddings=test_card_embeddings,
            max_selections=2,
            temperature=temp
        )
        print(f"Temperature {temp}: {temp_selection}")
    
    print("\nAction Head module testing complete!")