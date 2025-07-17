"""
Complete Balatro Policy with Straight-Through Estimation.

This module implements the full policy that handles both action selection
and conditional card parameter selection using straight-through estimation
for gradient flow through discrete choices.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from torch.distributions import Categorical

# Import our components
from text_encoder import make_text_encoder_module, ENCODER_DIM, CardEncoder
from lstm_module import make_lstm_module, HIDDEN_DIM
from action_head import (make_action_head_module, make_value_head_module, make_combinatorial_card_selector, 
                        ActionHead, ValueHead, ACTION_DIM)
from controller import Actions, format_card


class STECardSelector(nn.Module):
    """
    Card selector using straight-through estimation for differentiable discrete sampling.
    """
    
    def __init__(self, base_selector):
        """
        Initialize STE wrapper around base card selector.
        
        Args:
            base_selector: CombinatorialCardSelector instance
        """
        super().__init__()
        self.base_selector = base_selector
        self._device = base_selector._device
        
        # Learnable EOS token embedding
        self.eos_embedding = nn.Parameter(
            torch.randn(ENCODER_DIM, device=self._device) * 0.1
        )
    
    def forward_with_ste(
        self,
        context_embeddings: torch.Tensor,
        card_embeddings: torch.Tensor,
        max_selections: int = 5,
        temperature: float = 1.0,
        training: bool = True
    ) -> Tuple[List[int], torch.Tensor]:
        """
        Forward pass with straight-through estimation.
        
        Args:
            context_embeddings: Context from LSTM [batch_size, context_size]
            card_embeddings: Available card embeddings [num_cards, card_embedding_size]
            max_selections: Maximum number of cards to select
            temperature: Temperature for sampling
            training: Whether in training mode (uses STE) or inference mode
            
        Returns:
            Tuple of (selected_indices, total_log_prob)
        """
        if card_embeddings.size(0) == 0:
            return [], torch.tensor(0.0, device=self._device)
        
        batch_size = context_embeddings.size(0) if context_embeddings.dim() > 1 else 1
        
        # Handle single sample case
        if context_embeddings.dim() == 1:
            context_embeddings = context_embeddings.unsqueeze(0)
        
        # Project context to card embedding space
        initial_input = self.base_selector.context_projection(context_embeddings)
        initial_input = initial_input.unsqueeze(1)  # [batch_size, 1, card_embedding_size]
        
        # Initialize LSTM hidden state
        h_0 = torch.zeros(self.base_selector.num_layers, batch_size, self.base_selector.hidden_size, device=self._device)
        c_0 = torch.zeros(self.base_selector.num_layers, batch_size, self.base_selector.hidden_size, device=self._device)
        
        # Get initial LSTM output from context
        lstm_out, (h_n, c_n) = self.base_selector.combinatorial_lstm(initial_input, (h_0, c_0))
        
        selected_indices = []
        total_log_prob = torch.tensor(0.0, device=self._device)
        available_mask = torch.ones(card_embeddings.size(0), dtype=torch.bool, device=self._device)
        
        for step in range(min(max_selections, card_embeddings.size(0))):
            # Project LSTM output to card embedding space
            projected_output = self.base_selector.output_projection(lstm_out[:, -1, :])
            
            # Compute similarities with available cards
            card_similarities = torch.matmul(projected_output, card_embeddings.T)
            
            # Compute similarity with EOS token
            eos_similarity = torch.matmul(projected_output, self.eos_embedding.unsqueeze(-1))
            
            # Combine card similarities with EOS similarity
            # Shape: [batch_size, num_cards + 1] where last index is EOS
            all_similarities = torch.cat([card_similarities, eos_similarity], dim=-1)
            
            # Create extended mask (cards + EOS token - EOS is always available)
            extended_mask = torch.cat([
                available_mask, 
                torch.tensor([True], device=self._device)
            ])
            
            # Apply temperature scaling and mask
            logits = all_similarities / temperature
            logits = logits.masked_fill(~extended_mask.unsqueeze(0), float('-inf'))
            
            # Check if any cards are available (EOS is always available)
            if available_mask.sum() == 0 and step > 0:  # Allow at least one selection
                break
            
            # Get probabilities
            probs = F.softmax(logits, dim=-1)
            
            if training:
                # Straight-Through Estimator
                # Forward pass: discrete sampling
                hard_sample = torch.multinomial(probs.squeeze(0), num_samples=1)
                
                # Backward pass: continuous approximation  
                soft_sample = probs.argmax(dim=-1, keepdim=True)
                
                # STE: discrete forward, continuous backward
                selected_idx_tensor = hard_sample.float() + (soft_sample.squeeze(-1).float() - soft_sample.squeeze(-1).float().detach())
                selected_idx = hard_sample.item()
            else:
                # Inference: just sample discretely
                selected_idx = torch.multinomial(probs.squeeze(0), num_samples=1).item()
                selected_idx_tensor = torch.tensor(selected_idx, dtype=torch.float32, device=self._device)
            
            # Compute log probability for policy gradient
            log_prob = F.log_softmax(logits, dim=-1).squeeze(0)[selected_idx]
            total_log_prob = total_log_prob + log_prob
            
            # Check if EOS token was selected
            eos_idx = card_embeddings.size(0)  # EOS is at index num_cards
            if selected_idx == eos_idx:
                # EOS selected - terminate selection
                break
            
            # Add to selected indices (only if not EOS)
            selected_indices.append(selected_idx)
            
            # Update availability mask (only for valid card indices)
            if selected_idx < card_embeddings.size(0):
                available_mask[selected_idx] = False
            
            # Prepare next LSTM input (selected card embedding or EOS)
            if training:
                # Use STE version for gradient flow
                # Create one-hot with STE for all options (cards + EOS)
                all_embeddings = torch.cat([card_embeddings, self.eos_embedding.unsqueeze(0)], dim=0)
                one_hot = torch.zeros(all_embeddings.size(0), device=self._device)
                one_hot[selected_idx] = 1.0
                
                # Soft version for backward pass
                soft_one_hot = F.softmax(logits.squeeze(0), dim=-1)
                
                # STE one-hot
                ste_one_hot = one_hot + (soft_one_hot - soft_one_hot.detach())
                
                # Get embedding using STE weights
                selected_embedding = torch.matmul(ste_one_hot, all_embeddings)
                selected_embedding = selected_embedding.unsqueeze(0).unsqueeze(0)
            else:
                # Just use the actual selected embedding
                if selected_idx < card_embeddings.size(0):
                    selected_embedding = card_embeddings[selected_idx].unsqueeze(0).unsqueeze(0)
                else:
                    selected_embedding = self.eos_embedding.unsqueeze(0).unsqueeze(0)
            
            # Expand to batch size if needed
            if batch_size > 1:
                selected_embedding = selected_embedding.expand(batch_size, 1, -1)
            
            # Continue LSTM
            lstm_out, (h_n, c_n) = self.base_selector.combinatorial_lstm(selected_embedding, (h_n, c_n))
        print(f"In STECardSelector.forward_with_ste: selected indices: {selected_indices}, total log prob: {total_log_prob.item()}")
        return selected_indices, total_log_prob


class CompleteBalatroPolicy(TensorDictModule):
    """
    Complete Balatro policy handling action selection and conditional card selection.
    """
    
    # Actions that require card selection
    CARD_REQUIRING_ACTIONS = {
        Actions.PLAY_HAND.value,
        Actions.DISCARD_HAND.value, 
        Actions.BUY_CARD.value,
        Actions.SELL_JOKER.value,
        Actions.SELL_CONSUMABLE.value,
        Actions.SELECT_BOOSTER_CARD.value,
        Actions.USE_CONSUMABLE.value,
    }
    
    def __init__(
        self,
        device: str = "auto",
        temperature: float = 1.0
    ):
        """
        Initialize the complete policy.
        
        Args:
            device: Device to place modules on
            temperature: Temperature for card selection sampling
        """
        # Set device
        if device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)
        
        self.temperature = temperature
        
        # Create core components
        text_encoder = make_text_encoder_module(device=device)
        lstm = make_lstm_module(device=device)
        
        # Create custom action and value heads that expect "embeddings" (LSTM output key)
        from tensordict.nn import TensorDictModule as TDM
        
        # Action embedding layer - explicit action embeddings for STE
        self.action_embedding_dim = ENCODER_DIM  # Same as card embeddings for compatibility
        
        
        # Action head outputs continuous action vector (same dimension as action embeddings)
        action_net = ActionHead(input_size=HIDDEN_DIM, device=device)
        # Modify action head to output embeddings instead of logits
        action_net.action_projection = nn.Linear(HIDDEN_DIM, self.action_embedding_dim, device=self._device)
        action_head = TDM(action_net, in_keys=["embeddings"], out_keys=["action_vector"])
        
        value_net = ValueHead(input_size=HIDDEN_DIM, device=device) 
        value_head = TDM(value_net, in_keys=["embeddings"], out_keys=["state_value"])
        
        # Context projection for combined LSTM + action embeddings
        combined_context_dim = HIDDEN_DIM + self.action_embedding_dim  # 256 + 512 = 768
        context_projection = nn.Linear(combined_context_dim, HIDDEN_DIM, device=self._device)
        
        # Card selection components
        base_card_selector = make_combinatorial_card_selector(device=device)
        card_selector = STECardSelector(base_card_selector)

        card_encoder = CardEncoder(text_encoder.module)

        # Core pipeline for action vector and values
        core_pipeline = TensorDictSequential(
            text_encoder,
            lstm,
            action_head,
            value_head,
        )
        

        # Initialize parent
        super().__init__(
            module=core_pipeline,
            in_keys=["observation"],
            out_keys=["action_vector", "state_value"]
        )

        self.lstm = lstm
        self.text_encoder = text_encoder
        self.action_head = action_head
        self.value_head = value_head
        self.card_selector = card_selector
        self.core_pipeline = core_pipeline
        self.card_encoder = card_encoder
        self.action_embeddings = nn.Embedding(ACTION_DIM, self.action_embedding_dim, device=self._device)
        self.context_projection = context_projection
        # Move to device
        self.to(self._device)
    
    def forward(self, tensordict: TensorDict) -> TensorDict:
        """
        Forward pass through core pipeline.
        
        Args:
            tensordict: Input with observation and card embeddings
            
        Returns:
            TensorDict with action vector and state value
        """
        return self.core_pipeline(tensordict)
    
    def sample_action_and_cards(
        self,
        tensordict: TensorDict,
        deterministic: bool = False
    ) -> TensorDict:
        """
        Sample complete action including card selections.
        
        Args:
            tensordict: Input with observation and card embeddings
            deterministic: Whether to use deterministic (argmax) sampling
            
        Returns:
            TensorDict with sampled action, cards, and log probabilities
        """
        # Get core outputs
        core_output = self.forward(tensordict)
        
        # Sample primary action using STE with embeddings
        mask = tensordict.get("mask")
        print(f"Action mask: {mask}")

        # Get continuous action vector from policy head
        action_vector = core_output["action_vector"]  # [batch_size, action_embedding_dim]
        
        # Compute similarities with all action embeddings 
        action_similarities = torch.matmul(action_vector, self.action_embeddings.weight.T)  # [batch_size, num_actions]
        action_logits = action_similarities  # These are our logits
        action_logits = action_logits.masked_fill(~mask, float('-inf'))  # Apply action mask
        
        if deterministic:
            action = torch.argmax(action_logits, dim=-1)
            action_log_prob = F.log_softmax(action_logits, dim=-1)[action]
        else:
            action_dist = Categorical(logits=action_logits)
            print(f"Action distribution: {action_dist.probs}")
            action = action_dist.sample()
            action_log_prob = action_dist.log_prob(action)

        # Get quantized action embedding using STE
        if self.training:
            # Forward: discrete action embedding
            discrete_action_embedding = self.action_embeddings(action)
            # Backward: continuous action vector  
            action_embedding_ste = discrete_action_embedding + (action_vector - action_vector.detach())
        else:
            # Inference: just use discrete embedding
            action_embedding_ste = self.action_embeddings(action)
        
        # Initialize outputs
        selected_cards = []
        card_log_prob = torch.tensor(0.0, device=self._device)
        
        #from here on, switch to 1-indexed actions
        action = action + 1  # Convert to 1-indexed action

        # Conditional card selection using STE action embedding as context
        if action.item() in self.CARD_REQUIRING_ACTIONS:
            card_embeddings = self._get_card_embeddings_for_action(action.item(), tensordict)

            if card_embeddings is not None and card_embeddings.size(0) > 0:
                max_cards = self._get_max_cards_for_action(action.item())
                
                # Combine LSTM context with action embedding for full context
                lstm_context = core_output["embeddings"]  # Full game state context
                combined_context = torch.cat([lstm_context, action_embedding_ste], dim=-1)
                projected_context = self.context_projection(combined_context)  # Project to expected size
                
                selected_cards, card_log_prob = self.card_selector.forward_with_ste(
                    context_embeddings=projected_context,  # Full context + action embedding (projected)
                    card_embeddings=card_embeddings,
                    max_selections=max_cards,
                    temperature=self.temperature if not deterministic else 0.1,
                    training=self.training
                )
            else:
                print(f"No card embeddings available for action {Actions(action.item()).name}")
                selected_cards = []  # No cards selected if none available
        else:
            print(f"Action {Actions(action.item()).name} does not require card selection")
            selected_cards = []
        
        # Package outputs
        output = core_output.clone()
        output["action"] = action
        output["selected_cards"] = selected_cards
        output["action_log_prob"] = action_log_prob
        output["card_log_prob"] = card_log_prob
        output["total_log_prob"] = action_log_prob + card_log_prob
        output["logits"] = action_logits  # Add logits for RL compatibility
        
        print("In CompleteBalatroPolicy.sample_action_and_cards:")
        print(f"Sampled action: {action.item()} ({Actions(action.item()).name})")
        print(f"Selected cards: {selected_cards}")

        return output
    
    def _get_card_embeddings_for_action(self, action_idx: int, tensordict: TensorDict) -> Optional[torch.Tensor]:
        """
        Get relevant card embeddings for the given action.
        
        Args:
            action_idx: Index of the action
            tensordict: Input tensordict with card embeddings
            
        Returns:
            Card embeddings tensor or None if not applicable
        """
        action_type = Actions(action_idx)
        print(f"In _get_card_embeddings_for_action: action_type = {action_type}")
        
        if action_type == Actions.PLAY_HAND:
            return self._get_hand_card_embeddings(tensordict)
        elif action_type == Actions.DISCARD_HAND:
            return self._get_hand_card_embeddings(tensordict)
        elif action_type == Actions.BUY_CARD:
            return tensordict.get("shop_card_embeddings")
        elif action_type == Actions.SELL_JOKER:
            return tensordict.get("joker_embeddings")
        elif action_type == Actions.SELL_CONSUMABLE:
            return tensordict.get("consumable_embeddings")
        elif action_type == Actions.SELECT_BOOSTER_CARD:
            return tensordict.get("booster_card_embeddings")
        elif action_type == Actions.USE_CONSUMABLE:
            return tensordict.get("consumable_embeddings")
        
        return None

    def _get_hand_card_embeddings(self, tensordict: TensorDict) -> Optional[torch.Tensor]:
        """
        Get hand card embeddings from the tensordict.
        
        Args:
            tensordict: Input tensordict
            
        Returns:
            Hand card embeddings tensor or None if not available
        """
        # Get the full gamestate from the nested structure (wrapped in NonTensorData)
        full_gamestate_wrapper = tensordict.get("observation", {}).get("full_gamestate")
        print(f"In _get_hand_card_embeddings: full_gamestate_wrapper = {full_gamestate_wrapper}")
        
        if full_gamestate_wrapper is None:
            return torch.empty(0, ENCODER_DIM, device=self._device)
        
        # Extract the actual gamestate dictionary from NonTensorData
        full_gamestate = full_gamestate_wrapper.data if hasattr(full_gamestate_wrapper, 'data') else full_gamestate_wrapper
        print(f"In _get_hand_card_embeddings: actual full_gamestate = {full_gamestate}")
        
        hand = full_gamestate.get("hand", []) if isinstance(full_gamestate, dict) else []
        if hand is None or len(hand) == 0:
            return torch.empty(0, ENCODER_DIM, device=self._device)
        
        formatted_hand = []
        for card in hand:
            formatted_hand.append(format_card(card))
        embeddings = self.card_encoder(formatted_hand)
        return embeddings
    
    
    def _get_max_cards_for_action(self, action_idx: int) -> int:
        """
        Get maximum number of cards that can be selected for this action.
        
        Args:
            action_idx: Index of the action
            
        Returns:
            Maximum number of cards to select
        """
        action_type = Actions(action_idx)
        
        if action_type == Actions.PLAY_HAND:
            return 5  # Maximum poker hand size
        elif action_type == Actions.DISCARD_HAND:
            return 5  # Can discard multiple cards
        elif action_type in [Actions.BUY_CARD, Actions.SELL_JOKER, Actions.SELL_CONSUMABLE, 
                           Actions.SELECT_BOOSTER_CARD, Actions.USE_CONSUMABLE]:
            return 1  # Single card selection
        
        return 1  # Default to single selection


def make_complete_balatro_policy(
    device: str = "auto",
    temperature: float = 1.0
) -> CompleteBalatroPolicy:
    """
    Factory function to create a complete Balatro policy.
    
    Args:
        device: Device to place the policy on
        temperature: Temperature for card selection sampling
        
    Returns:
        CompleteBalatroPolicy instance
    """
    return CompleteBalatroPolicy(
        device=device,
        temperature=temperature
    )


if __name__ == "__main__":
    # Test the complete policy
    print("Testing Complete Balatro Policy with STE...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create policy
    policy = make_complete_balatro_policy(device=device, temperature=1.0)
    
    # Test observation
    test_observation = "Game State: Ante 1, Round 1. Hand: Ace of Spades, King of Hearts. Chips: 100."
    
    # Mock card embeddings
    hand_cards = torch.randn(2, ENCODER_DIM, device=device)  # 2 cards in hand
    shop_cards = torch.randn(3, ENCODER_DIM, device=device)  # 3 cards in shop
    jokers = torch.randn(1, ENCODER_DIM, device=device)      # 1 joker
    
    # Create input tensordict
    input_td = TensorDict({
        "observation": [test_observation],
        "full_gamestate": [{}],  # Empty dict for testing
        "hand_card_embeddings": hand_cards,
        "shop_card_embeddings": shop_cards, 
        "joker_embeddings": jokers,
        "is_init": torch.tensor(True, device=device)
    }, batch_size=[], device=device)
    
    print("Input observation:", test_observation[:50] + "...")
    print("Hand cards shape:", hand_cards.shape)
    print("Shop cards shape:", shop_cards.shape)
    
    # Test core forward pass
    print("\n=== Testing Core Forward Pass ===")
    core_output = policy.forward(input_td)
    print("Core output keys:", list(core_output.keys()))
    print("Action logits shape:", core_output["logits"].shape)
    print("State value shape:", core_output["state_value"].shape)
    
    # Test action sampling
    print("\n=== Testing Action Sampling ===")
    for deterministic in [False, True]:
        print(f"\nDeterministic: {deterministic}")
        
        action_output = policy.sample_action_and_cards(input_td, deterministic=deterministic)
        
        action = action_output["action"].item()
        selected_cards = action_output["selected_cards"]
        action_log_prob = action_output["action_log_prob"].item()
        card_log_prob = action_output["card_log_prob"].item()
        
        print(f"Sampled action: {action} ({Actions(action).name})")
        print(f"Selected cards: {selected_cards}")
        print(f"Action log prob: {action_log_prob:.4f}")
        print(f"Card log prob: {card_log_prob:.4f}")
        print(f"Total log prob: {action_output['total_log_prob'].item():.4f}")
    
    # Test specific card-requiring actions
    print("\n=== Testing Card-Requiring Actions ===")
    card_requiring_actions = [
        (Actions.PLAY_HAND.value, "hand_card_embeddings"),
        (Actions.BUY_CARD.value, "shop_card_embeddings"), 
        (Actions.SELL_JOKER.value, "joker_embeddings"),
    ]
    
    for action_idx, card_key in card_requiring_actions:
        print(f"\nForcing action: {Actions(action_idx).name}")
        
        # Create modified tensordict with forced action
        modified_td = input_td.clone()
        
        # Force the action by creating a one-hot logits tensor
        forced_logits = torch.full((21,), float('-inf'), device=device)
        forced_logits[action_idx] = 0.0  # Only this action is possible
        
        # Temporarily modify the policy to return these forced logits
        with torch.no_grad():
            core_output = policy.forward(modified_td)
            core_output["logits"] = forced_logits
            
            # Sample with forced action
            action_dist = Categorical(logits=forced_logits)
            action = action_dist.sample()
            action_log_prob = action_dist.log_prob(action)
            
            # Test card selection for this specific action
            card_embeddings = modified_td.get(card_key)
            print(f"Available cards: {card_embeddings.shape[0] if card_embeddings is not None else 0}")
            
            if card_embeddings is not None and card_embeddings.size(0) > 0:
                max_cards = policy._get_max_cards_for_action(action.item())
                print(f"Max cards for this action: {max_cards}")
                
                selected_cards, card_log_prob = policy.card_selector.forward_with_ste(
                    context_embeddings=core_output["embeddings"],
                    card_embeddings=card_embeddings,
                    max_selections=max_cards,
                    temperature=1.0,
                    training=True  # Test STE
                )
                
                print(f"Selected cards: {selected_cards}")
                print(f"Card log prob: {card_log_prob.item():.4f}")
                print(f"Number of cards selected: {len(selected_cards)}")
                print(f"Early termination: {len(selected_cards) < max_cards and len(selected_cards) < card_embeddings.shape[0]}")
            else:
                print("No card embeddings available for this action")
    
    # Test training mode vs eval mode
    print("\n=== Testing Training vs Eval Mode ===")
    
    # Training mode
    policy.train()
    train_output = policy.sample_action_and_cards(input_td)
    print("Training mode - selected cards:", train_output["selected_cards"])
    
    # Eval mode  
    policy.eval()
    eval_output = policy.sample_action_and_cards(input_td)
    print("Eval mode - selected cards:", eval_output["selected_cards"])
    
    # Test gradient flow
    print("\n=== Testing Gradient Flow ===")
    policy.train()
    policy.zero_grad()
    
    sample_output = policy.sample_action_and_cards(input_td)
    loss = sample_output["total_log_prob"]  # Dummy loss
    loss.backward()
    
    # Check if gradients exist
    has_gradients = any(p.grad is not None for p in policy.parameters())
    print(f"Gradients computed: {has_gradients}")
    
    if has_gradients:
        total_grad_norm = torch.sqrt(sum(p.grad.norm()**2 for p in policy.parameters() if p.grad is not None))
        print(f"Total gradient norm: {total_grad_norm:.6f}")
    
    print("\nComplete Balatro Policy testing complete!")