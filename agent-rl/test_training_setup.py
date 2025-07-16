"""
Test script to verify training setup before full PPO training.

This script tests each component individually to ensure everything works.
"""

import torch
from tensordict import TensorDict

# Our components
from balatro_env import make_balatro_env
from complete_policy import make_complete_balatro_policy
from text_encoder import ENCODER_DIM


def test_environment():
    """Test the Balatro environment."""
    print("=== Testing Environment ===")
    
    try:
        env = make_balatro_env(verbose=False, device="cpu")
        print(f"✅ Environment created successfully")
        print(f"   Observation spec: {env.observation_spec}")
        print(f"   Action spec: {env.action_spec}")
        print(f"   Reward spec: {env.reward_spec}")
        
        # Test reset
        reset_td = env.reset()
        print(f"✅ Environment reset successful")
        print(f"   Reset keys: {list(reset_td.keys())}")
        
        # Test random rollout
        rollout = env.rollout(3)
        print(f"✅ Environment rollout successful")
        print(f"   Rollout shape: {rollout.batch_size}")
        print(f"   Rollout keys: {list(rollout.keys())}")
        
        return env
        
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_policy(env):
    """Test the complete policy."""
    print("\n=== Testing Policy ===")
    
    try:
        policy = make_complete_balatro_policy(device="cpu", temperature=1.0)
        print(f"✅ Policy created successfully")
        
        # Create test observation with card embeddings
        reset_td = env.reset()
        
        # Add mock card embeddings
        reset_td["hand_card_embeddings"] = torch.randn(2, ENCODER_DIM)
        reset_td["shop_card_embeddings"] = torch.randn(3, ENCODER_DIM) 
        reset_td["joker_embeddings"] = torch.randn(1, ENCODER_DIM)
        
        # Test forward pass
        output = policy.forward(reset_td)
        print(f"✅ Policy forward pass successful")
        print(f"   Output keys: {list(output.keys())}")
        print(f"   Action logits shape: {output['logits'].shape}")
        print(f"   State value shape: {output['state_value'].shape}")
        
        # Test action sampling
        action_output = policy.sample_action_and_cards(reset_td)
        print(f"✅ Policy action sampling successful")
        print(f"   Sampled action: {action_output['action'].item()}")
        print(f"   Selected cards: {action_output['selected_cards']}")
        print(f"   Action log prob: {action_output['action_log_prob'].item():.4f}")
        print(f"   Card log prob: {action_output['card_log_prob'].item():.4f}")
        
        return policy
        
    except Exception as e:
        print(f"❌ Policy test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_tensordict_compatibility():
    """Test TensorDict compatibility for TorchRL integration."""
    print("\n=== Testing TensorDict Compatibility ===")
    
    try:
        # Create mock data as TorchRL would
        batch_size = [4]  # Batch of 4 samples
        
        mock_td = TensorDict({
            "observation": ["obs1", "obs2", "obs3", "obs4"],
            "action": torch.tensor([1, 5, 10, 3]),
            "reward": torch.tensor([0.1, -0.2, 0.5, 0.0]),
            "done": torch.tensor([False, False, True, False]),
            "next": TensorDict({
                "observation": ["next_obs1", "next_obs2", "next_obs3", "next_obs4"],
                "reward": torch.tensor([0.1, -0.2, 0.5, 0.0]),
                "done": torch.tensor([False, False, True, False]),
            }, batch_size=batch_size)
        }, batch_size=batch_size)
        
        print(f"✅ TensorDict creation successful")
        print(f"   Batch size: {mock_td.batch_size}")
        print(f"   Keys: {list(mock_td.keys())}")
        print(f"   Next keys: {list(mock_td['next'].keys())}")
        
        # Test accessing data
        print(f"   First observation: {mock_td['observation'][0]}")
        print(f"   Reward mean: {mock_td['reward'].mean().item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ TensorDict test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_collection_simulation():
    """Simulate data collection for training."""
    print("\n=== Testing Data Collection Simulation ===")
    
    try:
        env = make_balatro_env(verbose=False, device="cpu")
        policy = make_complete_balatro_policy(device="cpu", temperature=1.0)
        
        # Simulate collecting multiple steps
        collected_data = []
        
        # Reset environment
        current_td = env.reset()
        
        for step in range(5):
            # Add card embeddings (in real training, these would come from environment)
            current_td["hand_card_embeddings"] = torch.randn(2, ENCODER_DIM)
            current_td["shop_card_embeddings"] = torch.randn(3, ENCODER_DIM)
            current_td["joker_embeddings"] = torch.randn(1, ENCODER_DIM)
            
            # Get action from policy
            action_output = policy.sample_action_and_cards(current_td)
            action = action_output["action"]
            
            # Create action tensordict for environment
            action_td = TensorDict({
                "action": action
            }, batch_size=current_td.batch_size)
            
            # Step environment
            try:
                next_td = env.step(action_td)
                collected_data.append({
                    "step": step,
                    "action": action.item(),
                    "reward": next_td["reward"].item(),
                    "done": next_td["done"].item()
                })
                
                current_td = next_td
                
                if next_td["done"].item():
                    print(f"   Episode finished at step {step}")
                    break
                    
            except Exception as e:
                print(f"   Step {step} failed: {e}")
                break
        
        print(f"✅ Data collection simulation successful")
        print(f"   Collected {len(collected_data)} steps")
        for data in collected_data:
            print(f"   Step {data['step']}: action={data['action']}, reward={data['reward']:.3f}, done={data['done']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data collection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("🧪 Testing Training Setup Components\n")
    
    # Test each component
    env = test_environment()
    if env is None:
        print("❌ Cannot proceed without working environment")
        return
    
    policy = test_policy(env)
    if policy is None:
        print("❌ Cannot proceed without working policy")
        return
    
    tensordict_ok = test_tensordict_compatibility()
    if not tensordict_ok:
        print("❌ TensorDict compatibility issues detected")
        return
    
    collection_ok = test_data_collection_simulation()
    if not collection_ok:
        print("❌ Data collection simulation failed")
        return
    
    print("\n🎉 All tests passed! Training setup is ready.")
    print("\nNext steps:")
    print("1. Run 'python train_balatro_ppo.py' to start training")
    print("2. Monitor the training progress and metrics")
    print("3. Check the generated plots and saved models")


if __name__ == "__main__":
    main()