"""Temporal-Difference control: Q-Learning (off-policy) and SARSA (on-policy)."""

import numpy as np

from src.tabular.environment import MarioEnvironment
from src.tabular.policies import epsilon_greedy, greedy_policy


def train_q_learning(episodes=1200, alpha=0.10, gamma=0.95,
                     max_epsilon=0.10, min_epsilon=0.0, decay_rate=0.991):
    """
    Q-Learning is off-policy: the update always uses the BEST possible
    next action, regardless of which action the agent would actually pick
    under epsilon-greedy.

        Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]

    Returns:
        Qtable  (ndarray): learned Q-table, shape (25, 4)
        rewards (list):    total reward per episode
    """
    env = MarioEnvironment()
    Qtable = np.zeros((env.total_states, env.action_dimensions))
    epsilon = max_epsilon
    rewards = []

    for ep in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0

        while True:
            action = epsilon_greedy(state, Qtable, epsilon)
            next_state, reward, done = env.step(action)

            # Off-policy target: use the BEST next action
            best_next = greedy_policy(Qtable, next_state)
            td_target = reward if done else reward + gamma * Qtable[next_state, best_next]

            Qtable[state, action] += alpha * (td_target - Qtable[state, action])

            total_reward += reward
            state = next_state
            if done:
                break

        rewards.append(total_reward)
        epsilon = max(min_epsilon, epsilon * decay_rate)

    return Qtable, rewards


def train_sarsa(episodes=1200, alpha=0.10, gamma=0.95,
                max_epsilon=0.10, min_epsilon=0.0, decay_rate=0.991):
    """
    SARSA is on-policy: the update uses the action the agent ACTUALLY
    selects next (which may be exploratory).

        Q(s, a) <- Q(s, a) + alpha * [r + gamma * Q(s', a') - Q(s, a)]

    Because a' comes from the same epsilon-greedy policy, SARSA
    accounts for the risk of exploration. Near lava it learns lower
    values because it knows it might randomly slip in — making it
    more cautious than Q-Learning.

    Returns:
        Qtable  (ndarray): learned Q-table, shape (25, 4)
        rewards (list):    total reward per episode
    """
    env = MarioEnvironment()
    Qtable = np.zeros((env.total_states, env.action_dimensions))
    epsilon = max_epsilon
    rewards = []

    for ep in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0
        # SARSA needs the first action chosen BEFORE the loop starts
        action = epsilon_greedy(state, Qtable, epsilon)

        while True:
            next_state, reward, done = env.step(action)

            # On-policy target: use the ACTUAL next action
            next_action = epsilon_greedy(next_state, Qtable, epsilon)
            td_target = reward if done else reward + gamma * Qtable[next_state, next_action]

            Qtable[state, action] += alpha * (td_target - Qtable[state, action])

            total_reward += reward
            state = next_state
            action = next_action
            if done:
                break

        rewards.append(total_reward)
        epsilon = max(min_epsilon, epsilon * decay_rate)

    return Qtable, rewards
