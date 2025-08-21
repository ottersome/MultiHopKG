from numpy import common_type
from torch._C import _cuda_tunableop_set_max_tuning_duration
from multihopkg.utils.ops import int_fill_var_cuda, var_cuda, zeros_var_cuda
from multihopkg.utils import ops
import torch
from torch import nn
from typing import Tuple, Dict, List, Optional, Union
import pdb

import torch.nn.functional as F

import sys

def init_layer_uniform(layer: nn.Linear, init_w: float = 3e-3) -> nn.Linear:
    """Init uniform parameters on the single layer."""
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)
    return layer

class AttentionGraphNavigator(nn.Module):
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
        super(AttentionGraphNavigator, self).__init__()
        
        # Dimensions
        self.state_dim = state_dim
        self.target_dim = target_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Target encoder
        self.target_encoder = nn.Sequential(
            nn.Linear(target_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Attention mechanism to focus on relevant parts of state and target
        self.attention = MultiHeadAttention(hidden_dim, num_heads=4)
        
        # Action predictor
        self.action_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_dim * 2)  # Output both mu and log_sigma
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
        # Encode state and target
        state_encoded = self.state_encoder(state)  # (batch_size, hidden_dim)
        target_encoded = self.target_encoder(target)  # (batch_size, hidden_dim)
        
        # Apply attention between state and target
        attended_features = self.attention(
            state_encoded.unsqueeze(1),  # (batch_size, 1, hidden_dim)
            target_encoded.unsqueeze(1)   # (batch_size, 1, hidden_dim)
        )  # (batch_size, 1, hidden_dim)
        
        attended_features = attended_features.squeeze(1)  # (batch_size, hidden_dim)
        
        # Concatenate attended features with target encoding for action prediction
        combined = torch.cat([attended_features, target_encoded], dim=-1)  # (batch_size, hidden_dim*2)
        
        # Predict action parameters (mu and log_sigma)
        action_params = self.action_predictor(combined)  # (batch_size, action_dim*2)
        
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
                hidden_dim=dim_hidden
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

