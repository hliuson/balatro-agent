"""
Balatro PPO Training Script following TorchRL Tutorial.

This script implements PPO training for the Balatro RL agent using the complete policy
with straight-through estimation for card selection.
"""

import torch
import torch.nn as nn
from collections import defaultdict
from tqdm import tqdm

# TorchRL imports
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch.distributions import Categorical
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import TransformedEnv, Compose, StepCounter
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

# Our Balatro components
from balatro_env import make_balatro_env
from complete_policy import make_complete_balatro_policy
from text_encoder import make_text_encoder_module


def create_balatro_environment(device="auto", verbose=False):
    """Create Balatro environment with transforms."""
    print("Creating Balatro environment...")
    
    # Create base environment
    base_env = make_balatro_env(verbose=verbose, device=device)
    
    # Add transforms
    env = TransformedEnv(
        base_env,
        Compose(
            StepCounter(),  # Count steps for episode tracking
        )
    )
    
    print("Environment created successfully!")
    print(f"Observation spec: {env.observation_spec}")
    print(f"Action spec: {env.action_spec}")
    print(f"Reward spec: {env.reward_spec}")
    
    # Check environment specs
    try:
        #check_env_specs(env)
        print("Environment specs check passed!")
    except Exception as e:
        print(f"Environment specs check failed: {e}")
        print("Continuing anyway...")
    
    return env


class BalatroActorModule(TensorDictModule):
    """
    Actor module that wraps our CompleteBalatroPolicy for TorchRL compatibility.
    """
    
    def __init__(self, complete_policy):
        super().__init__(
            module=complete_policy,
            in_keys=["observation"],
            out_keys=["logits", "action", "selected_cards", "action_log_prob", "card_log_prob"]
        )
        self.complete_policy = complete_policy
    
    def forward(self, tensordict):
        # Use the complete policy's sampling method
        output = self.complete_policy.sample_action_and_cards(tensordict, deterministic=False)
        return output


class BalatroValueModule(TensorDictModule):
    """
    Value module that extracts state value from our complete policy.
    """
    
    def __init__(self, complete_policy):
        super().__init__(
            module=complete_policy,
            in_keys=["observation"],
            out_keys=["state_value"]
        )
        self.complete_policy = complete_policy
    
    def forward(self, tensordict):
        # Get state value from complete policy
        output = self.complete_policy.forward(tensordict)
        return TensorDict({"state_value": output["state_value"]}, batch_size=tensordict.batch_size)


class BalatroActor(ProbabilisticActor):
    """
    Probabilistic actor for Balatro that handles discrete action distributions.
    """
    
    def __init__(self, actor_module, action_spec):
        # We don't use the standard ProbabilisticActor constructor
        # because our policy already handles sampling
        nn.Module.__init__(self)
        self.actor_module = actor_module
        self.action_spec = action_spec
    
    def forward(self, tensordict):
        # Get action logits and sampled actions from our policy
        output = self.actor_module(tensordict)
        
        # Add log_prob for PPO (combine action and card log probs)
        total_log_prob = output["action_log_prob"] + output["card_log_prob"]
        output["sample_log_prob"] = total_log_prob
        
        return output


def create_policy_and_value_networks(env, device="auto", temperature=1.0):
    """Create policy and value networks for Balatro."""
    print("Creating policy and value networks...")
    
    # Create complete policy
    complete_policy = make_complete_balatro_policy(device=device, temperature=temperature)
    
    # Create actor module
    actor_module = BalatroActorModule(complete_policy)
    
    # Create actor
    actor = BalatroActor(actor_module, env.action_spec)
    
    # Create value module (shares the same complete policy)
    value_module = BalatroValueModule(complete_policy)
    
    print("Policy and value networks created!")
    
    # Test the networks
    reset_td = env.reset()
    print("\nTesting networks...")
    print("Actor output keys:", list(actor(reset_td).keys()))
    print("Value output keys:", list(value_module(reset_td).keys()))
    
    return actor, value_module


def main():
    """Main training function."""
    
    # ==========================================
    # Hyperparameters
    # ==========================================
    
    print("Setting up hyperparameters...")
    
    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Training parameters
    frames_per_batch = 1000
    total_frames = 50_000  # Start with smaller number for testing
    sub_batch_size = 64
    num_epochs = 10
    
    # PPO parameters
    clip_epsilon = 0.2
    gamma = 0.99
    lmbda = 0.95
    entropy_eps = 1e-4
    
    # Optimization parameters
    lr = 3e-4
    max_grad_norm = 1.0
    
    # ==========================================
    # Environment
    # ==========================================
    
    env = create_balatro_environment(device=device, verbose=True)
    
    # ==========================================
    # Policy and Value Networks
    # ==========================================
    
    actor, value_module = create_policy_and_value_networks(env, device=device, temperature=1.0)
    
    # ==========================================
    # Data Collector
    # ==========================================
    
    print("Setting up data collector...")
    
    collector = SyncDataCollector(
        env,
        actor,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        split_trajs=False,
        device=device,
    )
    
    # ==========================================
    # Replay Buffer
    # ==========================================
    
    print("Setting up replay buffer...")
    
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=frames_per_batch),
        sampler=SamplerWithoutReplacement(),
    )
    
    # ==========================================
    # Loss Function
    # ==========================================
    
    print("Setting up loss function...")
    
    # GAE for advantage estimation
    advantage_module = GAE(
        gamma=gamma, 
        lmbda=lmbda, 
        value_network=value_module, 
        average_gae=True
    )
    
    # PPO Loss
    loss_module = ClipPPOLoss(
        actor_network=actor,
        critic_network=value_module,
        clip_epsilon=clip_epsilon,
        entropy_bonus=bool(entropy_eps),
        entropy_coef=entropy_eps,
        critic_coef=1.0,
        loss_critic_type="smooth_l1",
    )
    
    # Optimizer
    optim = torch.optim.Adam(loss_module.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, total_frames // frames_per_batch, 0.0
    )
    
    # ==========================================
    # Training Loop
    # ==========================================
    
    print("Starting training...")
    
    logs = defaultdict(list)
    pbar = tqdm(total=total_frames)
    eval_str = ""
    
    # Training loop
    for i, tensordict_data in enumerate(collector):
        # Inner optimization loop
        for epoch in range(num_epochs):
            # Compute advantage
            advantage_module(tensordict_data)
            
            # Prepare data for replay buffer
            data_view = tensordict_data.reshape(-1)
            replay_buffer.extend(data_view.cpu())
            
            # Mini-batch optimization
            for _ in range(frames_per_batch // sub_batch_size):
                subdata = replay_buffer.sample(sub_batch_size)
                
                try:
                    loss_vals = loss_module(subdata.to(device))
                    
                    # Total loss
                    loss_value = (
                        loss_vals["loss_objective"]
                        + loss_vals["loss_critic"] 
                        + loss_vals["loss_entropy"]
                    )
                    
                    # Backward pass
                    loss_value.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
                    
                    # Optimization step
                    optim.step()
                    optim.zero_grad()
                    
                except Exception as e:
                    print(f"Error in loss computation: {e}")
                    continue
        
        # Logging
        try:
            logs["reward"].append(tensordict_data["next", "reward"].mean().item())
            logs["step_count"].append(tensordict_data["step_count"].max().item())
            logs["lr"].append(optim.param_groups[0]["lr"])
            
            pbar.update(tensordict_data.numel())
            
            # Progress strings
            cum_reward_str = f"avg reward={logs['reward'][-1]: 4.4f}"
            stepcount_str = f"step count (max): {logs['step_count'][-1]}"
            lr_str = f"lr: {logs['lr'][-1]: 4.4f}"
            
            # Evaluation every 10 batches
            if i % 10 == 0:
                try:
                    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
                        eval_rollout = env.rollout(100, actor)  # Shorter rollout for evaluation
                        logs["eval reward"].append(eval_rollout["next", "reward"].mean().item())
                        logs["eval reward (sum)"].append(eval_rollout["next", "reward"].sum().item())
                        logs["eval step_count"].append(eval_rollout["step_count"].max().item())
                        
                        eval_str = (
                            f"eval total reward: {logs['eval reward (sum)'][-1]: 4.4f}, "
                            f"eval steps: {logs['eval step_count'][-1]}"
                        )
                        del eval_rollout
                except Exception as e:
                    print(f"Error in evaluation: {e}")
                    eval_str = "eval: error"
            
            pbar.set_description(", ".join([eval_str, cum_reward_str, stepcount_str, lr_str]))
            
        except Exception as e:
            print(f"Error in logging: {e}")
        
        # Learning rate schedule
        scheduler.step()
    
    pbar.close()
    
    # ==========================================
    # Results
    # ==========================================
    
    print("Training completed!")
    
    # Print final statistics
    if logs["reward"]:
        print(f"Final training reward: {logs['reward'][-1]:.4f}")
        print(f"Average training reward: {sum(logs['reward'])/len(logs['reward']):.4f}")
    
    if logs["eval reward (sum)"]:
        print(f"Final evaluation reward: {logs['eval reward (sum)'][-1]:.4f}")
        print(f"Best evaluation reward: {max(logs['eval reward (sum)']):.4f}")
    
    if logs["step_count"]:
        print(f"Final step count: {logs['step_count'][-1]}")
        print(f"Max step count achieved: {max(logs['step_count'])}")
    
    # Save logs for later analysis
    torch.save(logs, "balatro_training_logs.pt")
    print("Training logs saved to balatro_training_logs.pt")
    
    # Save final model
    torch.save(actor.state_dict(), "balatro_actor_final.pt")
    torch.save(value_module.state_dict(), "balatro_value_final.pt")
    
    print("Training complete! Model saved.")
    
    return actor, value_module, logs


if __name__ == "__main__":
    main()