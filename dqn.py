# dqn.py
# This file defines the Deep Q-Network (DQN) model used in reinforcement learning.
# It supports standard DQN, as well as Dueling DQN architectures.

import torch
from torch import nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, enable_dueling_dqn=True):
        """
        Initialize the DQN model.

        Args:
            state_dim (int): Dimension of the state/input vector.
            action_dim (int): Number of possible discrete actions.
            hidden_dim (int): Number of neurons in the first hidden layer.
            enable_dueling_dqn (bool): If True, use dueling DQN architecture.
        """
        super(DQN, self).__init__()
        self.enable_dueling_dqn = enable_dueling_dqn

        # First fully connected layer (common to all architectures)
        self.fc1 = nn.Linear(state_dim, hidden_dim)

        if self.enable_dueling_dqn:
            # Dueling DQN splits into two separate streams after the first layer:
            # One for state-value (V) and another for advantages (A).

            # Value stream layers
            self.fc_value = nn.Linear(hidden_dim, 256)  # Hidden layer for value
            self.value = nn.Linear(256, 1)  # Outputs a scalar V(s)

            # Advantage stream layers
            self.fc_advantages = nn.Linear(hidden_dim, 256)  # Hidden layer for advantages
            self.advantages = nn.Linear(256, action_dim)  # Outputs A(s, a) for each action

        else:
            # Standard DQN: single stream that outputs Q-values directly
            self.output = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (Tensor): Input state tensor of shape (batch_size, state_dim)

        Returns:
            Tensor: Q-values for each action.
        """
        # Common first layer with ReLU activation
        x = F.relu(self.fc1(x))

        if self.enable_dueling_dqn:
            # Value stream
            v = F.relu(self.fc_value(x))  # Hidden layer
            V = self.value(v)  # Output V(s), shape: (batch_size, 1)

            # Advantage stream
            a = F.relu(self.fc_advantages(x))  # Hidden layer
            A = self.advantages(a)  # Output A(s, a), shape: (batch_size, num_actions)

            # Combine value and advantage into Q-values:
            # Q(s, a) = V(s) + (A(s, a) - mean(A(s, ·)))
            # Subtracting the mean ensures identifiability and improves stability.
            Q = V + A - torch.mean(A, dim=1, keepdim=True)
        else:
            # Simple DQN: directly output Q-values
            Q = self.output(x)

        return Q
