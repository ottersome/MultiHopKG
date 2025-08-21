from numpy import common_type
from torch._C import _cuda_tunableop_set_max_tuning_duration
from multihopkg.utils.ops import int_fill_var_cuda, var_cuda, zeros_var_cuda
from multihopkg.utils import ops
import torch
from torch import nn
from typing import Tuple, Dict, List, Optional, Union
import pdb

import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

import sys

def init_layer_uniform(layer: nn.Linear, init_w: float = 3e-3) -> nn.Linear:
    """Init uniform parameters on the single layer."""
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)
    return layer

class TransformerBlock(nn.Module):
    """A single transformer block with multi-head attention and feed-forward network."""
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super(TransformerBlock, self).__init__()
        
        self.attention = MultiHeadAttention(hidden_dim, num_heads)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual connection
        attn_out = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Feed-forward with residual connection
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x

class AttentionGraphNavigator(nn.Module):
    """
    A GPT-like transformer network that learns to navigate through a TransE graph.
    Takes a current state and a target and produces actions to navigate towards the target.
    """
    def __init__(
        self,
        state_dim: int,
        target_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super(AttentionGraphNavigator, self).__init__()
        
        # Dimensions
        self.state_dim = state_dim
        self.target_dim = target_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Input projections (replace separate encoders with simple projections)
        self.state_proj = nn.Linear(state_dim, hidden_dim)
        self.target_proj = nn.Linear(target_dim, hidden_dim)
        
        # Multi-layer transformer blocks (like GPT)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(hidden_dim)
        
        # Action predictor
        self.action_predictor = nn.Linear(hidden_dim, action_dim * 2)  # Output both mu and log_sigma
        
    def forward(self, state: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the navigator.
        
        Args:
            state: Current state embedding (batch_size, state_dim)
            target: Target embedding (batch_size, target_dim)
            
        Returns:
            action_params: Action parameters (batch_size, action_dim*2)
                          First half is mu, second half is log_sigma
        """
        # Project inputs to hidden dimension
        state_proj = self.state_proj(state).unsqueeze(1)  # (batch_size, 1, hidden_dim)
        target_proj = self.target_proj(target).unsqueeze(1)  # (batch_size, 1, hidden_dim)
        
        # Concatenate state and target as sequence
        x = torch.cat([state_proj, target_proj], dim=1)  # (batch_size, 2, hidden_dim)
        
        # Pass through transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        
        # Apply final layer norm
        x = self.final_norm(x)
        
        # Pool the sequence (take mean of state and target representations)
        pooled = x.mean(dim=1)  # (batch_size, hidden_dim)
        
        # Predict action parameters (mu and log_sigma)
        action_params = self.action_predictor(pooled)  # (batch_size, action_dim*2)
        
        return action_params

class SimpleGraphNavigator(nn.Module):
    """
    A neural network that learns to navigate through a TransE graph.
    Takes a current state and a target (either a node or a question embedding)
    and produces actions to navigate towards the target.
    """
    def __init__(
        self,
        state_dim: int,
        target_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super(SimpleGraphNavigator, self).__init__()
        
        # Dimensions
        self.state_dim = state_dim
        self.target_dim = target_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # State encoder
        self.combined_parameter_estimation = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        
    def forward(self, state: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the navigator.
        
        Args:
            state: Current state embedding (batch_size, state_dim)
            target: Target embedding (batch_size, target_dim)
            
        Returns:
            action_params: Action parameters (batch_size, action_dim*2)
                          First half is mu, second half is log_sigma
        """
        action_params = self.combined_parameter_estimation(torch.cat([state, target], dim=-1))  # (batch_size, hidden_dim*2)
        
        
        return action_params

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism to focus on relevant parts of the input.
    """
    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super(MultiHeadAttention, self).__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: Optional[torch.Tensor] = None):
        """
        Forward pass of multi-head attention.
        
        Args:
            query: Query tensor (batch_size, seq_len_q, hidden_dim)
            key: Key tensor (batch_size, seq_len_k, hidden_dim)
            value: Value tensor (batch_size, seq_len_k, hidden_dim), defaults to key if None
            
        Returns:
            output: Attention output (batch_size, seq_len_q, hidden_dim)
        """
        if value is None:
            value = key
            
        batch_size = query.size(0)
        
        # Project inputs to queries, keys, and values
        q = self.q_proj(query)  # (batch_size, seq_len_q, hidden_dim)
        k = self.k_proj(key)    # (batch_size, seq_len_k, hidden_dim)
        v = self.v_proj(value)  # (batch_size, seq_len_k, hidden_dim)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len_q, head_dim)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len_k, head_dim)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len_k, head_dim)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (batch_size, num_heads, seq_len_q, seq_len_k)
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)  # (batch_size, num_heads, seq_len_q, seq_len_k)
        
        # Apply attention weights to values
        context = torch.matmul(attn_weights, v)  # (batch_size, num_heads, seq_len_q, head_dim)
        
        # Reshape back
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)  # (batch_size, seq_len_q, hidden_dim)
        
        # Apply output projection
        output = self.out_proj(context)  # (batch_size, seq_len_q, hidden_dim)
        
        return output

class AttentionContinuousPolicyGradient(nn.Module):
    def __init__(
        self,
        beta: float,
        gamma: float,
        dim_action: int,
        dim_hidden: int,
        dim_observation: int,
        use_attention: bool = True,
        log_std_min: float = -20,
        log_std_max: float = -2,
        use_tanh_squashing: bool= True,  # Make squashing configurable
    ):
        super().__init__()
        
        self.beta = beta
        self.gamma = gamma
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.use_tanh_squashing = use_tanh_squashing
        self.use_attention = use_attention
        
        if self.use_attention:
            self.navigator = AttentionGraphNavigator(
                state_dim=dim_observation // 2,  # Assuming observation is [state, target] concatenated
                target_dim=dim_action,
                action_dim=dim_action,
                hidden_dim=dim_hidden,
                num_layers=4,  # More layers like GPT
                num_heads=8    # More attention heads
            )
        else:
            self.navigator = SimpleGraphNavigator(
                state_dim=dim_observation // 2,  # Assuming observation is [state, target] concatenated
                target_dim=dim_action,
                action_dim=dim_action,
                hidden_dim=dim_hidden
            )

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def get_beta(self):
        return self.beta
    
    def get_gamma(self):
        return self.gamma

    def forward(
        self, observations: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._sample_action_with_navigator(observations, target)

    def _sample_action_with_navigator(
        self,
        observations: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        action_params = self.navigator(observations, target)
        
        # DEBUGGING: REMOVE This comment afteward
        action_dim = action_params.size(-1) // 2
        mu = action_params[:, :action_dim]
        log_sigma_raw = action_params[:, action_dim:]

        # Apply log_std bounds for numerical stability
        log_sigma = torch.clamp(log_sigma_raw, self.log_std_min, self.log_std_max)
        sigma = torch.exp(log_sigma)
         
        # Create distribution
        dist = torch.distributions.Normal(mu, sigma)
        entropy = dist.entropy().sum(dim=-1)
         
        # Sample action
        z = dist.rsample()
        
        if self.use_tanh_squashing:
            # Apply tanh squashing for bounded actions
            actions = torch.tanh(z)
            # Correct log probabilities for squashing
            log_probs = dist.log_prob(z) - torch.log(1 - actions.pow(2) + 1e-7)
            log_probs = log_probs.sum(dim=-1)
        else:
            actions = z
            log_probs = dist.log_prob(actions).sum(dim=-1)
        
        return actions, log_probs, entropy, mu, sigma

class ContinuousPolicyGradient(nn.Module):
    # TODO: remove all parameters that are irrelevant here
    def __init__(
        # TODO: remove all parameters that are irrelevant here
        self,
        beta: float,
        gamma: float,
        dim_action: int,
        dim_hidden: int,
        dim_observation: int,
        log_std_min: float = -20,
        log_std_max: float = 2,
    ):
        super(ContinuousPolicyGradient, self).__init__()

        # Training hyperparameters
        self.beta = beta  # entropy regularization parameter
        self.gamma = gamma  # shrinking factor

        ########################################
        # Torch Modules
        ########################################
        self.hidden1, self.hidden2, self.mu_layer, self.sigma_layer = self._define_modules(
            input_dim=dim_observation, observation_dim=dim_action, hidden_dim=dim_hidden
        )

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def get_beta(self):
        return self.beta
    
    def get_gamma(self):
        return self.gamma

    def forward(
        self, observations: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Once we do the observations we need to do the sampling
        return self._sample_action(observations)

    def _sample_action(
        self,
        observations: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Will sample batch_len actions given batch_len observations
        args
            observations: torch.Tensor. Shape: (batch_len, path_encoder_dim)
        """
        projections = F.relu(self.hidden1(observations))
        projections = F.relu(self.hidden2(projections))

        mu = self.mu_layer(projections).tanh()

        log_sigma = self.sigma_layer(projections).tanh()  # Stabilizing Sigma, Preventing extremes
        log_sigma = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_sigma + 1)

        sigma = torch.exp(log_sigma)

        # # Create a normal distribution using the mean and standard deviation
        dist = torch.distributions.Normal(mu, sigma)
        entropy = dist.entropy().sum(dim=-1)

        # # Now Sample from it 
        # # TODO: Ensure we are sampling correctly from this 
        z = dist.rsample()

        # normalize action and log_prob
        # see appendix C of https://arxiv.org/abs/1801.01290
        # Trick to ensure the actions are within the range [-1, 1]
        actions = z.tanh()
        log_probs = dist.log_prob(z) - torch.log(1 - actions.pow(2) + 1e-7)
        log_probs = log_probs.sum(-1)

        # log_probs = dist.log_prob(actions).sum(dim=-1) # ONLY WORKS ASSUMING INDEPENDENCE (which so far we obey).

        return actions, log_probs, entropy, mu, sigma


    def _define_modules(self, input_dim:int, observation_dim: int, hidden_dim: int):

        hidden1 = nn.Linear(input_dim, hidden_dim)
        hidden2 = nn.Linear(hidden_dim, hidden_dim)
        
        mu_layer = nn.Linear(hidden_dim, observation_dim)
        sigma_layer = nn.Linear(hidden_dim, observation_dim)

        # Custom initialization
        mu_layer = init_layer_uniform(mu_layer)
        sigma_layer = init_layer_uniform(sigma_layer)

        return hidden1, hidden2, mu_layer, sigma_layer

    def _reparemeteriztion(self, dist, action):
        return dist.log_prob(action).sum(dim=-1)


class SACGraphNavigator(nn.Module):
    """
    SAC implementation for graph navigation with optional supervised auxiliary loss.
    """
    def __init__(
        self,
        beta: float,
        gamma: float,
        dim_action: int,
        dim_hidden: int,
        dim_observation: int,
        use_attention: bool = True,
        log_std_min: float = -20,
        log_std_max: float = 2,
        use_tanh_squashing: bool = True,
        lr: float = 3e-4,
        tau: float = 0.005,
        use_auxiliary_loss: bool = False,
        auxiliary_loss_weight: float = 0.1,
    ):
        super().__init__()
        
        self.gamma = gamma
        self.tau = tau
        self.use_auxiliary_loss = use_auxiliary_loss
        self.auxiliary_loss_weight = auxiliary_loss_weight
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Actor (Policy Network)
        if use_attention:
            self.actor = AttentionContinuousPolicyGradient(
                beta=beta,
                gamma=gamma,
                dim_action=dim_action,
                dim_hidden=dim_hidden,
                dim_observation=dim_observation,
                use_attention=use_attention,
                log_std_min=log_std_min,
                log_std_max=log_std_max,
                use_tanh_squashing=use_tanh_squashing,
            )
        else:
            self.actor = ContinuousPolicyGradient(
                beta=beta,
                gamma=gamma,
                dim_action=dim_action,
                dim_hidden=dim_hidden,
                dim_observation=dim_observation,
                log_std_min=log_std_min,
                log_std_max=log_std_max,
            )
        
        # Critics (Q-Networks)
        self.critic_q1 = self._build_critic(dim_observation + dim_action, dim_hidden)
        self.critic_q2 = self._build_critic(dim_observation + dim_action, dim_hidden)
        
        # Target Critics
        self.critic_q1_target = self._build_critic(dim_observation + dim_action, dim_hidden)
        self.critic_q2_target = self._build_critic(dim_observation + dim_action, dim_hidden)
        
        # Initialize target networks
        self.critic_q1_target.load_state_dict(self.critic_q1.state_dict())
        self.critic_q2_target.load_state_dict(self.critic_q2.state_dict())
        
        # Automatic entropy tuning
        self.target_entropy = -dim_action
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_q1_optimizer = optim.Adam(self.critic_q1.parameters(), lr=lr)
        self.critic_q2_optimizer = optim.Adam(self.critic_q2.parameters(), lr=lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        
    def _build_critic(self, input_dim: int, hidden_dim: int) -> nn.Module:
        """Build a critic network."""
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, observations: torch.Tensor, target: torch.Tensor = None):
        """Forward pass through actor."""
        if hasattr(self.actor, 'navigator'):  # Attention-based
            return self.actor(observations, target)
        else:  # Simple policy gradient
            return self.actor(observations)
    
    def get_q_values(self, observations: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get Q-values from both critics."""
        obs_action = torch.cat([observations, actions], dim=-1)
        q1 = self.critic_q1(obs_action)
        q2 = self.critic_q2(obs_action)
        return q1, q2
    
    def get_target_q_values(self, observations: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get Q-values from target critics."""
        obs_action = torch.cat([observations, actions], dim=-1)
        q1_target = self.critic_q1_target(obs_action)
        q2_target = self.critic_q2_target(obs_action)
        return q1_target, q2_target
    
    def update_critics(
        self, 
        observations: torch.Tensor, 
        actions: torch.Tensor, 
        rewards: torch.Tensor, 
        next_observations: torch.Tensor, 
        next_targets: torch.Tensor,
        dones: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update critic networks."""
        with torch.no_grad():
            # Get next actions from current policy
            if hasattr(self.actor, 'navigator'):
                next_actions, next_log_probs, _, _, _ = self.actor(next_observations, next_targets)
            else:
                next_actions, next_log_probs, _, _, _ = self.actor(next_observations)
            
            # Get target Q-values
            next_q1_target, next_q2_target = self.get_target_q_values(next_observations, next_actions)
            next_q_target = torch.min(next_q1_target, next_q2_target)
            
            # Compute target
            alpha = self.log_alpha.exp()
            target_q = rewards + self.gamma * (1 - dones) * (next_q_target - alpha * next_log_probs.unsqueeze(-1))
        
        # Get current Q-values
        current_q1, current_q2 = self.get_q_values(observations, actions)
        
        # Compute critic losses
        critic_q1_loss = F.mse_loss(current_q1, target_q)
        critic_q2_loss = F.mse_loss(current_q2, target_q)
        
        # Update critics
        self.critic_q1_optimizer.zero_grad()
        critic_q1_loss.backward()
        self.critic_q1_optimizer.step()
        
        self.critic_q2_optimizer.zero_grad()
        critic_q2_loss.backward()
        self.critic_q2_optimizer.step()
        
        return critic_q1_loss, critic_q2_loss
    
    def update_actor_and_alpha(
        self, 
        observations: torch.Tensor, 
        targets: torch.Tensor = None,
        optimal_actions: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Update actor network and alpha (temperature parameter)."""
        # Get actions from current policy
        if hasattr(self.actor, 'navigator'):
            actions, log_probs, _, mu, _ = self.actor(observations, targets)
        else:
            actions, log_probs, _, mu, _ = self.actor(observations)
        
        # Get Q-values for current actions
        q1, q2 = self.get_q_values(observations, actions)
        min_q = torch.min(q1, q2)
        
        # Actor loss (SAC objective)
        alpha = self.log_alpha.exp()
        actor_loss = (alpha * log_probs.unsqueeze(-1) - min_q).mean()
        
        # Add supervised auxiliary loss if enabled
        auxiliary_loss = torch.tensor(0.0, device=self.device)
        if self.use_auxiliary_loss and optimal_actions is not None:
            auxiliary_loss = F.mse_loss(mu, optimal_actions)
            total_actor_loss = actor_loss + self.auxiliary_loss_weight * auxiliary_loss
        else:
            total_actor_loss = actor_loss
        
        # Update actor
        self.actor_optimizer.zero_grad()
        total_actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update alpha (temperature parameter)
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        return actor_loss, alpha_loss, auxiliary_loss
    
    def soft_update_targets(self):
        """Soft update of target networks."""
        for target_param, param in zip(self.critic_q1_target.parameters(), self.critic_q1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.critic_q2_target.parameters(), self.critic_q2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def compute_optimal_actions(self, current_states: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute optimal actions for TransE navigation.
        In TransE space, optimal action is approximately target - current_state.
        """
        return targets - current_states


class ReplayBuffer:
    """Experience replay buffer for SAC."""
    def __init__(self, capacity: int, obs_dim: int, action_dim: int):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    def push(self, obs, action, reward, next_obs, next_target, done):
        """Add experience to buffer."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (obs, action, reward, next_obs, next_target, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int):
        """Sample batch of experiences."""
        import random
        batch = random.sample(self.buffer, batch_size)
        obs, action, reward, next_obs, next_target, done = map(torch.stack, zip(*batch))
        return obs, action, reward, next_obs, next_target, done
    
    def __len__(self):
        return len(self.buffer)

