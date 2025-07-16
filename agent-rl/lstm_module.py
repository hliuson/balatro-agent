"""
LSTM Module for Balatro TorchRL Agent.

This module provides TorchRL-compatible LSTM functionality for the policy.
It uses TorchRL's LSTMModule for proper state management and batch optimization.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.modules import LSTMModule, set_recurrent_mode
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec

# Model architecture constants
ENCODER_DIM = 512
HIDDEN_DIM = 256
NUM_LAYERS = 2


def make_lstm_module(
    input_size: int = ENCODER_DIM,
    hidden_size: int = HIDDEN_DIM,
    num_layers: int = NUM_LAYERS,
    device: str = "auto"
) -> LSTMModule:
    """
    Factory function to create a TorchRL LSTMModule.
    
    Args:
        input_size: Size of input embeddings
        hidden_size: Size of LSTM hidden state
        num_layers: Number of LSTM layers
        device: Device to place the module on
        
    Returns:
        LSTMModule configured for TorchRL
    """
    # Set device
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    # Create LSTM module
    lstm_module = LSTMModule(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        in_key="embeddings",
        out_key="embeddings",  # Output overwrites input
        device=device
    )
    
    return lstm_module


class CustomLSTMModule(TensorDictModule):
    """
    Custom LSTM module that wraps PyTorch LSTM for TorchRL.
    
    This provides more control over the LSTM behavior while still
    being compatible with TorchRL's TensorDict interface.
    """
    
    def __init__(
        self,
        input_size: int = ENCODER_DIM,
        hidden_size: int = HIDDEN_DIM,
        num_layers: int = NUM_LAYERS,
        device: str = "auto"
    ):
        """
        Initialize custom LSTM module.
        
        Args:
            input_size: Size of input embeddings
            hidden_size: Size of LSTM hidden state
            num_layers: Number of LSTM layers
            device: Device to place the module on
        """
        # Set device
        if device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)
        
        # Create LSTM
        lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        ).to(self._device)
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Initialize as TensorDictModule
        super().__init__(
            module=lstm,
            in_keys=["embeddings", "recurrent_state_h", "recurrent_state_c"],
            out_keys=["lstm_embeddings", "recurrent_state_h", "recurrent_state_c"]
        )
        
        # Store reference to LSTM for custom forward
        self.lstm = lstm
    
    def forward(self, tensordict: TensorDict) -> TensorDict:
        """
        Forward pass through LSTM.
        
        Args:
            tensordict: Input TensorDict with embeddings and optional states
            
        Returns:
            TensorDict with updated embeddings and states
        """
        # Get embeddings
        embeddings = tensordict["embeddings"]
        
        # Handle batch dimensions
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0).unsqueeze(0)  # [1, 1, input_size]
        elif embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(1)  # [batch, 1, input_size]
        
        batch_size = embeddings.size(0)
        
        # Get or initialize hidden states
        if "recurrent_state_h" in tensordict and "recurrent_state_c" in tensordict:
            h_0 = tensordict["recurrent_state_h"]
            c_0 = tensordict["recurrent_state_c"]
        else:
            h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self._device)
            c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self._device)
        
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(embeddings, (h_0, c_0))
        
        # Update tensordict
        tensordict = tensordict.clone()
        tensordict["lstm_embeddings"] = lstm_out.squeeze(1)  # Remove sequence dimension
        tensordict["recurrent_state_h"] = h_n
        tensordict["recurrent_state_c"] = c_n
        
        return tensordict
    
    def make_tensordict_primer(self) -> TensorDictModule:
        """
        Create a TensorDict primer for initializing LSTM states.
        
        Returns:
            TensorDictModule that initializes recurrent states
        """
        class StateInitializer(nn.Module):
            def __init__(self, hidden_size: int, num_layers: int, device: torch.device):
                super().__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                self.device = device
            
            def forward(self, batch_size: torch.Size = torch.Size([])) -> Dict[str, torch.Tensor]:
                if len(batch_size) == 0:
                    batch_size = torch.Size([1])
                
                return {
                    "recurrent_state_h": torch.zeros(
                        self.num_layers, batch_size[0], self.hidden_size, device=self.device
                    ),
                    "recurrent_state_c": torch.zeros(
                        self.num_layers, batch_size[0], self.hidden_size, device=self.device
                    )
                }
        
        initializer = StateInitializer(self.hidden_size, self.num_layers, self._device)
        
        return TensorDictModule(
            module=initializer,
            in_keys=[],
            out_keys=["recurrent_state_h", "recurrent_state_c"]
        )


def make_custom_lstm_module(
    input_size: int = ENCODER_DIM,
    hidden_size: int = HIDDEN_DIM,
    num_layers: int = NUM_LAYERS,
    device: str = "auto"
) -> CustomLSTMModule:
    """
    Factory function to create a custom LSTM module.
    
    Args:
        input_size: Size of input embeddings
        hidden_size: Size of LSTM hidden state
        num_layers: Number of LSTM layers
        device: Device to place the module on
        
    Returns:
        CustomLSTMModule configured for TorchRL
    """
    return CustomLSTMModule(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        device=device
    )


if __name__ == "__main__":
    # Test the LSTM modules
    print("Testing LSTM modules...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Test TorchRL's LSTMModule
    print("\n=== Testing TorchRL LSTMModule ===")
    lstm_module = make_lstm_module(device=device)
    
    # Test input
    test_embeddings = torch.randn(ENCODER_DIM, device=device)
    
    # Create TensorDict input
    input_td = TensorDict({
        "embeddings": test_embeddings,
        "is_init": torch.tensor(True, device=device)  # TorchRL LSTM needs this
    }, batch_size=[], device=device)
    
    print("Input embeddings shape:", input_td["embeddings"].shape)
    
    # Forward pass
    output_td = lstm_module(input_td)
    print("Output keys:", list(output_td.keys()))
    print("Output embeddings shape:", output_td["embeddings"].shape)
    if "next" in output_td:
        print("Next keys:", list(output_td["next"].keys()))
        if "recurrent_state_h" in output_td["next"]:
            print("Hidden state shape:", output_td["next"]["recurrent_state_h"].shape)
        if "recurrent_state_c" in output_td["next"]:
            print("Cell state shape:", output_td["next"]["recurrent_state_c"].shape)
    
    # Test with recurrent mode using context manager
    print("\n=== Testing Recurrent Mode ===")
    
    # Test sequence input
    seq_embeddings = torch.randn(1, 5, ENCODER_DIM, device=device)  # [batch, seq, features]
    seq_input_td = TensorDict({
        "embeddings": seq_embeddings,
        "is_init": torch.tensor([[True, False, False, False, False]], device=device)  # Shape [1, 5] for sequence
    }, batch_size=[1], device=device)
    
    print("Sequence embeddings shape:", seq_input_td["embeddings"].shape)
    
    # Use context manager for recurrent mode
    with set_recurrent_mode(True):
        seq_output_td = lstm_module(seq_input_td)
    print("Sequence output embeddings shape:", seq_output_td["embeddings"].shape)
    
    # Test Custom LSTM Module
    print("\n=== Testing Custom LSTM Module ===")
    custom_lstm = make_custom_lstm_module(device=device)
    
    # Test single embedding
    custom_input_td = TensorDict({
        "embeddings": test_embeddings
    }, batch_size=[], device=device)
    
    print("Custom input embeddings shape:", custom_input_td["embeddings"].shape)
    custom_output_td = custom_lstm(custom_input_td)
    print("Custom output keys:", list(custom_output_td.keys()))
    print("Custom output embeddings shape:", custom_output_td["lstm_embeddings"].shape)
    print("Custom hidden state shape:", custom_output_td["recurrent_state_h"].shape)
    print("Custom cell state shape:", custom_output_td["recurrent_state_c"].shape)
    
    # Test state persistence
    print("\n=== Testing State Persistence ===")
    # Create new input with same embeddings but previous states
    second_input_td = TensorDict({
        "embeddings": test_embeddings,  # Use original embeddings
        "recurrent_state_h": custom_output_td["recurrent_state_h"],
        "recurrent_state_c": custom_output_td["recurrent_state_c"]
    }, batch_size=[], device=device)
    
    second_output_td = custom_lstm(second_input_td)
    print("Second step output embeddings shape:", second_output_td["lstm_embeddings"].shape)
    
    # Test primer
    print("\n=== Testing State Primer ===")
    primer = custom_lstm.make_tensordict_primer()
    primer_output = primer(TensorDict({}, batch_size=[]))
    print("Primer output keys:", list(primer_output.keys()))
    print("Primer hidden state shape:", primer_output["recurrent_state_h"].shape)
    print("Primer cell state shape:", primer_output["recurrent_state_c"].shape)
    
    print("\nLSTM module testing complete!")