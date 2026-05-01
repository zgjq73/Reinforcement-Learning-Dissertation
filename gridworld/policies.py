"""Action selection policies for tabular Q-learning methods."""

# greedy policy and epislon greedy policy functions

import random
import numpy as np


def greedy_policy(table, state):
    """
    Pure greedy selection — always picks the action with the highest Q-value.

    Parameters:
        table (ndarray): Q-table, shape (n_states, n_actions)
        state (int):     current state

    Returns:
        int: the best action
    """
    return int(np.argmax(table[state]))


def epsilon_greedy(state, table, epsilon):
    """
    Epsilon-greedy action selection.

    With probability `epsilon` pick a random action (exploration);
    otherwise pick the greedy best action (exploitation).

    Parameters:
        state   (int):     current state
        table   (ndarray): Q-table, shape (n_states, n_actions)
        epsilon (float):   probability of choosing randomly

    Returns:
        int: chosen action
    """
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, table.shape[1] - 1)
    return greedy_policy(table, state)
