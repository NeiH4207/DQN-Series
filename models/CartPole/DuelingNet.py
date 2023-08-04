#!/usr/bin/env python
import numpy as np
import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch import optim as optim
from models.GymDQN import GymDQN

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
        self.layer2 = nn.Linear(128, 128)
        self.value = nn.Linear(128, n_actions)
        self.advance = nn.Linear(128, n_actions)
        self.set_optimizer(optimizer, lr)
        self.loss_history = np.array([])

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        value = self.value(x) 
        adv = self.advance(x) 
        mean_adv = torch.mean(adv, dim=1, keepdim=True)
        Q = value + adv - mean_adv
        return Q 