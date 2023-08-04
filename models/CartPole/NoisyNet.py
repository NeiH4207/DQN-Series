#!/usr/bin/env python
import numpy as np
import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch import optim as optim
from models.GymDQN import GymDQN
from models.NoisyLayer import NoisyLinear

class CartPole(GymDQN):

    def __init__(
        self, 
        n_observations: int, 
        n_actions: int, 
        optimizer: str = "adamw",
        lr: float = 0.001) -> None:
        super(GymDQN, self).__init__()
        self.n_observations = n_observations
        self.n_actions = n_actions
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = NoisyLinear(128, 128)
        self.Q = NoisyLinear(128, n_actions)
        self.set_optimizer(optimizer, lr)
        self.loss_history = np.array([])

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.Q(x) 
    
    def reset_noise(self):
        """Reset all noisy layers."""
        self.layer2.reset_noise()
        self.Q.reset_noise()