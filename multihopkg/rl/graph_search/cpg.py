from numpy import common_type
from torch._C import _cuda_tunableop_set_max_tuning_duration
from multihopkg.utils.ops import int_fill_var_cuda, var_cuda, zeros_var_cuda
from multihopkg.utils import ops
import torch
from torch import nn
from typing import Tuple
import pdb

import torch.nn.functional as F

import sys

def init_layer_uniform(layer: nn.Linear, init_w: float = 3e-3) -> nn.Linear:
    """Init uniform parameters on the single layer."""
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)
    return layer

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

