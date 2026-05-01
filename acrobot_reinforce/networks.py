"""Policy and Value networks used by the REINFORCE-family agents."""

import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    """
    Maps a state to a probability distribution over actions.

    Used by REINFORCE, REINFORCE+Baseline, and Actor-Critic.
    """

    def __init__(self, state_dim, action_dim, hidden_size=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class ValueNetwork(nn.Module):
    """
    Maps a state to a scalar value V(s).

    Used as a baseline / critic by REINFORCE+Baseline and Actor-Critic.
    """

    def __init__(self, state_dim, hidden_size=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
