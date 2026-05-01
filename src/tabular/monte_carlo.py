"""First-Visit Monte Carlo control on the gridworld."""

import numpy as np

from src.tabular.environment import MarioEnvironment
from src.tabular.policies import epsilon_greedy


def run_episode(env, table, epsilon):
    """
    Run one full episode using epsilon-greedy and return the trajectory.

    Returns:
        list of (state, action, reward) tuples
    """
    trajectory = []
    state = env.reset()
    while True:
        action = epsilon_greedy(state, table, epsilon)
        next_state, reward, done = env.step(action)
        trajectory.append((state, action, reward))
        state = next_state
        if done:
            break
    return trajectory


def train_monte_carlo(episodes=10000, alpha=0.05, gamma=0.95,
                      max_epsilon=1.0, min_epsilon=0.01, decay_rate=0.999):
    """
    Train a First-Visit Monte Carlo agent on the gridworld.

    Returns:
        Qtable          (ndarray): learned Q-table, shape (25, 4)
        episode_rewards (list):    total reward per episode
    """
    env = MarioEnvironment()
    Qtable = np.zeros((env.total_states, env.action_dimensions))
    epsilon = max_epsilon
    episode_rewards = []

    for ep in range(1, episodes + 1):
        # Step 1: generate a full episode
        trajectory = run_episode(env, Qtable, epsilon)
        episode_rewards.append(sum(step[2] for step in trajectory))

        # Step 2: First-Visit MC update
        # Walk backward through the trajectory, accumulating the
        # discounted return G. Update Q(s, a) only on the FIRST
        # visit to each (state, action) pair.
        G = 0
        for t in range(len(trajectory) - 1, -1, -1):
            state, action, reward = trajectory[t]
            G = gamma * G + reward

            visited_before = any(
                trajectory[i][0] == state and trajectory[i][1] == action
                for i in range(t)
            )
            if not visited_before:
                Qtable[state, action] += alpha * (G - Qtable[state, action])

        # Step 3: decay exploration rate
        epsilon = max(min_epsilon, epsilon * decay_rate)

    return Qtable, episode_rewards
