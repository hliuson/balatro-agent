# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from typing import Any, Dict, List, Tuple, Union

from tqdm import tqdm

def _unstack_nested_dict(nested_dict: Dict[str, Any], env_idx: int) -> Dict[str, Any]:
    """Recursively unstack a nested dictionary at the given environment index."""
    result = {}
    for key, val in nested_dict.items():
        if isinstance(val, dict):
            # Recursively unstack nested dicts
            result[key] = _unstack_nested_dict(val, env_idx)
        elif isinstance(val, (list, tuple)) and len(val) > env_idx:
            # Unstack lists/tuples at environment index
            result[key] = val[env_idx]
        else:
            # Keep scalars or invalid indices as-is
            result[key] = val
    return result

def _stack_nested_values(values: List[Any]) -> Any:
    """Stack a list of values, handling nested dictionaries recursively."""
    if not values:
        return []
    
    first_val = values[0]
    
    # Handle nested dictionaries
    if isinstance(first_val, dict):
        stacked_dict = {}
        for key in first_val.keys():
            key_values = [val.get(key) for val in values if isinstance(val, dict)]
            stacked_dict[key] = _stack_nested_values(key_values)
        return stacked_dict
    
    # Handle tensors
    elif all(isinstance(v, torch.Tensor) for v in values):
        return torch.stack(values, dim=0)
    
    # Handle numpy arrays
    elif all(isinstance(v, np.ndarray) for v in values):
        return np.stack(values, axis=0)
    
    # For everything else (primitives, mixed types, etc.), just wrap in list
    else:
        return values

def unstack_dict_observations(obs_dict: Dict[str, Any], num_envs: int) -> List[Dict[str, Any]]:
    """
    Unstack a dict observation with vectorized values into a list of individual observations.
    Handles nested dictionaries recursively.
    
    Example:
    {a: [v1, v2], b: {c: [x1, x2]}} -> [{a: v1, b: {c: x1}}, {a: v2, b: {c: x2}}]
    """
    if not obs_dict:
        return []
    
    unstacked_obs = []
    for env_idx in range(num_envs):
        env_obs = {}
        for key, val in obs_dict.items():
            if isinstance(val, dict):
                # Handle nested dicts recursively
                env_obs[key] = _unstack_nested_dict(val, env_idx)
            elif isinstance(val, (list, tuple)) and len(val) > env_idx:
                # Unstack at environment index
                env_obs[key] = val[env_idx]
            else:
                # Keep scalars or invalid indices as-is
                env_obs[key] = val
        unstacked_obs.append(env_obs)
    
    return unstacked_obs

def stack_dict_observations(obs_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Stack a list of dict observations into a single dict with stacked values.
    Handles nested dictionaries recursively.
    
    Example:
    [{a: v1, b: {c: x1}}, {a: v2, b: {c: x2}}] -> {a: [v1, v2], b: {c: [x1, x2]}}
    """
    if not obs_list:
        return {}
    
    stacked_obs = {}
    # Get all keys from the first observation
    for key in obs_list[0].keys():
        values = [obs[key] for obs in obs_list]
        stacked_obs[key] = _stack_nested_values(values)
    
    return stacked_obs

def flatten_dict_observations(obs_list: List[Dict[str, Any]], indices: np.ndarray) -> List[Dict[str, Any]]:
    """
    Extract observations at specific indices from a list of dict observations.
    Returns a list of dict observations.
    """
    if not obs_list:
        return []
    
    return [obs_list[i] for i in indices]

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "balatro-rl"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "Balatro-v0"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 4
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if env_id == "Balatro-v0":
            from balatro_env import BalatroGymEnv
            env = BalatroGymEnv()
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        return env

    return thunk


class Agent(nn.Module):
    def __init__(self, envs, card_dim=64, hidden_size=512, lstm_layers=2):
        super().__init__()
        
        # Feature encoder for structured observations
        from feature_encoder import BalatroFeatureEncoder
        self.feature_encoder = BalatroFeatureEncoder(
            card_dim=card_dim,
            hidden_dim=hidden_size,
            num_transformer_layers=3,
            num_attention_heads=8
        )
        
        # Calculate input size: 6 component types * hidden_size + game_state_hidden
        encoder_output_size = hidden_size * 6 + hidden_size
        
        self.torso = nn.LSTM(
            input_size=encoder_output_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.0,
        )
        
        # For Dict action space, get the number of discrete actions from action_type
        if isinstance(envs.single_action_space, gym.spaces.Dict):
            num_actions = envs.single_action_space["action_type"].n
        else:
            num_actions = envs.single_action_space.n
            
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions),
        )

        # Pointer network for card selection - uses card embeddings directly
        self.pointer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, card_dim),  # Match card embedding dimension
        )

        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def encode_observation(self, observation: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Encode raw game state observation using feature encoder."""
        from feature_encoder import CardFeatureExtractor
        
        # Extract structured features from raw game state
        raw_game_state = observation["raw_game_state"]
        features = CardFeatureExtractor.extract_features(raw_game_state)
        
        # Convert to tensors
        device = next(self.parameters()).device
        
        cards = torch.tensor(features["cards"], dtype=torch.int32).to(device)
        card_types = torch.tensor(features["card_types"], dtype=torch.int32).to(device)
        source_types = torch.tensor(features["source_types"], dtype=torch.int32).to(device)
        game_state = torch.tensor(features["game_state"], dtype=torch.int32).to(device)
        
        # Create batch dimension if needed
        if cards.dim() == 2:
            cards = cards.unsqueeze(0)
            card_types = card_types.unsqueeze(0)
            source_types = source_types.unsqueeze(0)
            game_state = game_state.unsqueeze(0)
        
        # Encode using feature encoder
        encoded = self.feature_encoder({
            "cards": cards,
            "card_types": card_types,
            "source_types": source_types,
            "game_state": game_state
        })
        
        return encoded

    def get_value(self, observation: Dict[str, Any], lstm_hidden: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        encoded = self.encode_observation(observation)
        x = encoded["state_embed"].unsqueeze(1)  # Add sequence dimension
        x, _ = self.torso(x, lstm_hidden)
        return self.critic(x)

    def get_action_and_value(self, observation: Dict[str, Any], lstm_hidden: Tuple[torch.Tensor, torch.Tensor], action=None, card=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        encoded = self.encode_observation(observation)
        
        x = encoded["state_embed"].unsqueeze(1)  # Add sequence dimension
        x, new_lstm_hidden = self.torso(x, lstm_hidden)

        action_logits = self.actor(x).squeeze(1)
        
        action_masks = observation["action_mask"]
        if isinstance(action_masks, torch.Tensor):
            mask_tensor = action_masks.to(action_logits.device)
        elif isinstance(action_masks, np.ndarray):
            mask_tensor = torch.from_numpy(action_masks).bool().to(action_logits.device)
        else:
            mask_tensor = torch.tensor(action_masks, dtype=torch.bool, device=action_logits.device)

        action_logits = action_logits.masked_fill(~mask_tensor, -1e8)

        action_dist = Categorical(logits=action_logits)
        if action is None:
            action = action_dist.sample()
        action_logprob = action_dist.log_prob(action)
        actions_entropy = action_dist.entropy()

        value = self.critic(x).squeeze(1)
        
        # ---- POINTER NETWORK ----
        # Use context-aware card embeddings from feature encoder
        card_embeddings = encoded["card_embeddings"]  # [batch_size, max_cards, card_dim]
        batch_size = x.size(0)
        max_cards = card_embeddings.size(1)
        
        # Create query vector from LSTM output
        pointer_query = self.pointer(x)  # [batch_size, card_dim]
        
        # Compute attention over all cards
        if max_cards > 0:
            # Flatten card embeddings for batch processing
            pointer_logits = torch.matmul(pointer_query, card_embeddings.transpose(1, 2))  # [batch_size, max_cards]
            
            # Create mask for valid cards (non-padding)
            valid_mask = (encoded["card_types"] != 6)  # 6 is padding type
            pointer_logits = pointer_logits.masked_fill(~valid_mask, -1e8)
            
            pointer_dist = Categorical(logits=pointer_logits)
            
            if card is None:
                card = pointer_dist.sample()
            card_logprob = pointer_dist.log_prob(card)
            cards_entropy = pointer_dist.entropy()
        else:
            # No cards available
            card = torch.zeros(batch_size, dtype=torch.long, device=x.device)
            card_logprob = torch.zeros(batch_size, device=x.device)
            cards_entropy = torch.zeros(batch_size, device=x.device)

        action = action.squeeze()
        card = card.squeeze()
        value = value.squeeze()
        action_logprob = action_logprob.squeeze()
        card_logprob = card_logprob.squeeze()
        actions_entropy = actions_entropy.squeeze()
        cards_entropy = cards_entropy.squeeze()

        return value, action, action_logprob, actions_entropy, card, card_logprob, cards_entropy, new_lstm_hidden

    def get_original_idx(self, card_index: int, observation: Dict[str, Any]) -> Tuple[str, int]:
        """Map flat card index back to original source and index."""
        # Count total cards across all sources to find the original
        sources = ["hand", "joker", "consumable", "shop", "booster", "voucher"]
        source_names = ["hand_cards", "jokers", "consumables", "shop_items", "boosters", "vouchers"]
        
        current_idx = 0
        for source_idx, (source, source_name) in enumerate(zip(sources, source_names)):
            mask = observation["type_masks"][source]
            count = int(np.sum(mask))
            if card_index < current_idx + count:
                return source_name, card_index - current_idx + 1  # 1-based index
            current_idx += count
        
        return "hand_cards", 1  # Fallback

if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup - Modified for structured observations
    obs = [None] * args.num_steps  # Will store list of dict observations
    actions = torch.zeros((args.num_steps, args.num_envs)).to(device)
    cards = torch.zeros((args.num_steps, args.num_envs)).to(device)  # Card selections from pointer network
    action_logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    card_logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    
    # LSTM hidden states
    hidden_size = 512  # Should match agent's hidden_size
    lstm_layers = 2    # Should match agent's lstm_layers
    lstm_hidden = (
        torch.zeros(lstm_layers, args.num_envs, hidden_size).to(device),
        torch.zeros(lstm_layers, args.num_envs, hidden_size).to(device)
    )

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in tqdm(range(1, args.num_iterations + 1), desc="Training", unit="iter"):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in tqdm(range(0, args.num_steps), desc=f"Rollout {iteration}", unit="step", leave=False):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # Reset LSTM hidden states for terminated episodes
            if step > 0:
                # Reset hidden states where episodes ended
                reset_mask = dones[step - 1].bool()
                if reset_mask.any():
                    lstm_hidden[0][:, reset_mask] = 0
                    lstm_hidden[1][:, reset_mask] = 0

            # ALGO LOGIC: action logic
            with torch.no_grad():
                value, action, action_logprob, _, card, card_logprob, _, new_lstm_hidden = agent.get_action_and_value(next_obs, lstm_hidden)
                values[step] = value.flatten()
                lstm_hidden = new_lstm_hidden
                
            actions[step] = action
            cards[step] = card
            action_logprobs[step] = action_logprob
            card_logprobs[step] = card_logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            # Convert actions to environment format for vectorized environments
            action_types = []
            card_indices = []
            
            for i in range(args.num_envs):
                action_type = int(action[i].cpu().numpy())
                card_idx = int(card[i].cpu().numpy())
                
                # Convert flat card index to (source, index_in_source) format
                obs_for_env = {}
                for key, val in next_obs.items():
                    if isinstance(val, (list, tuple)) and len(val) > i:
                        obs_for_env[key] = val[i]
                    else:
                        obs_for_env[key] = val
                        
                card_source_str, card_index_in_source = agent.get_original_idx(card_idx, obs_for_env)
                
                # Map string to CardSources enum value
                source_mapping = {
                    "hand": 0,
                    "joker": 1,
                    "consumable": 2,
                    "shop": 3,
                    "booster": 4,
                    "voucher": 6,
                }
                card_source_value = source_mapping.get(card_source_str, 0)
                
                action_types.append(action_type + 1)
                card_indices.append(np.array([card_source_value, card_index_in_source], dtype=np.int32))
            
            # Structure actions for vectorized environment
            env_actions = {
                "action_type": action_types,
                "card_index": card_indices
            }
            next_obs, reward, terminations, truncations, infos = envs.step(env_actions)
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_done = torch.Tensor(next_done).to(device)

            for i in range(args.num_envs):
                if terminations[i]:
                    writer.add_scalar("charts/episodic_return", infos["episode_return"][i], global_step)
        print("Reached max rollout steps, calculating rewards and advantages")
        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs, lstm_hidden).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        print("Finished calculating advantages and returns")
        # flatten the batch - handle dict observations differently
        # First, flatten the observations from (num_steps, num_envs) to (batch_size,)
        b_obs = []
        for step in range(args.num_steps):
            step_obs = obs[step]  # This is a dict with vectorized values for all envs
            # Unstack the vectorized observation into individual observations
            individual_obs = unstack_dict_observations(step_obs, args.num_envs)
            b_obs.extend(individual_obs)
        
        b_action_logprobs = action_logprobs.reshape(-1)
        b_card_logprobs = card_logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_cards = cards.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        print("Training model from replay buffer")
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                # Extract minibatch observations from the flattened observations 
                mb_obs_list = flatten_dict_observations(b_obs, mb_inds)
                # Stack the list of dict observations into a single dict for the agent
                mb_obs = stack_dict_observations(mb_obs_list)
                
                # We need LSTM hidden states for training - reinitialize for each minibatch
                mb_lstm_hidden = (
                    torch.zeros(lstm_layers, len(mb_inds), hidden_size).to(device),
                    torch.zeros(lstm_layers, len(mb_inds), hidden_size).to(device)
                )
                
                newvalue, _, new_action_logprob, action_entropy, _, new_card_logprob, card_entropy, _ = agent.get_action_and_value(
                    mb_obs, mb_lstm_hidden, b_actions.long()[mb_inds], b_cards.long()[mb_inds]
                )
                
                # Compute ratios for both action and card policies (factorized joint policy)
                action_logratio = new_action_logprob - b_action_logprobs[mb_inds]
                card_logratio = new_card_logprob - b_card_logprobs[mb_inds]
                # Joint policy ratio = exp(log π_new(a) - log π_old(a) + log π_new(c) - log π_old(c))
                logratio = action_logratio + card_logratio
                ratio = logratio.exp()
                
                # Combined entropy
                entropy = action_entropy + card_entropy

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
