import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, layer_1_size=32, layer_2_size=64):
        """
        Build a fully connected neural network

        Parameters
        ----------
        state_size (int): State size
        action_size (int): Action size
        seed (int): random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.layer_1 = nn.Linear(state_size, layer_1_size)
        self.layer_2 = nn.Linear(layer_1_size, layer_2_size)
        self.op_layer = nn.Linear(layer_2_size, action_size)

    def forward(self, x):
        """Forward pass"""
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = self.op_layer(x)
        return x
