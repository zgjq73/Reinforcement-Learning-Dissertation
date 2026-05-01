"""
Temporal-Difference control: Q-Learning (off-policy) and SARSA (on-policy).

Run directly with:  python gridworld/td_methods.py
"""

import random
import numpy as np

from environment import MarioEnvironment
from policies import epsilon_greedy, greedy_policy


# ============================================================
# HYPERPARAMETERS
# ============================================================

SEED        = 42
EPISODES    = 1200
ALPHA       = 0.10
GAMMA       = 0.95
MAX_EPSILON = 0.10
MIN_EPSILON = 0.0
DECAY_RATE  = 0.991


def train_q_learning():
    """
    Q-Learning is off-policy: the update always uses the BEST possible
    next action, regardless of which action the agent would actually pick
    under epsilon-greedy.

        Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]
    """
    env = MarioEnvironment()
    Qtable = np.zeros((env.total_states, env.action_dimensions))
    epsilon = MAX_EPSILON
    rewards = []

    for ep in range(1, EPISODES + 1):
        state = env.reset()
        total_reward = 0

        while True:
            action = epsilon_greedy(state, Qtable, epsilon)
            next_state, reward, done = env.step(action)

            # Off-policy target: use the BEST next action
            best_next = greedy_policy(Qtable, next_state)
            td_target = reward if done else reward + GAMMA * Qtable[next_state, best_next]

            Qtable[state, action] += ALPHA * (td_target - Qtable[state, action])

            total_reward += reward
            state = next_state
            if done:
                break

        rewards.append(total_reward)
        epsilon = max(MIN_EPSILON, epsilon * DECAY_RATE)

    return Qtable, rewards


def train_sarsa():
    """
    SARSA is on-policy: the update uses the action the agent ACTUALLY
    selects next (which may be exploratory).

        Q(s, a) <- Q(s, a) + alpha * [r + gamma * Q(s', a') - Q(s, a)]

    Because a' comes from the same epsilon-greedy policy, SARSA
    accounts for the risk of exploration. Near lava it learns lower
    values because it knows it might randomly slip in - making it
    more cautious than Q-Learning.
    """
    env = MarioEnvironment()
    Qtable = np.zeros((env.total_states, env.action_dimensions))
    epsilon = MAX_EPSILON
    rewards = []

    for ep in range(1, EPISODES + 1):
        state = env.reset()
        total_reward = 0
        # SARSA needs the first action chosen BEFORE the loop starts
        action = epsilon_greedy(state, Qtable, epsilon)

        while True:
            next_state, reward, done = env.step(action)

            # On-policy target: use the ACTUAL next action
            next_action = epsilon_greedy(next_state, Qtable, epsilon)
            td_target = reward if done else reward + GAMMA * Qtable[next_state, next_action]

            Qtable[state, action] += ALPHA * (td_target - Qtable[state, action])

            total_reward += reward
            state = next_state
            action = next_action
            if done:
                break

        rewards.append(total_reward)
        epsilon = max(MIN_EPSILON, epsilon * DECAY_RATE)

    return Qtable, rewards


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    print("Training Q-Learning agent...")
    _, rewards_ql = train_q_learning()
    np.save("ql_rewards.npy", np.array(rewards_ql))
    print(f"Saved {len(rewards_ql)} episode rewards to ql_rewards.npy")

    # Re-seed before SARSA so it sees the same noise stream
    random.seed(SEED)
    np.random.seed(SEED)

    print("Training SARSA agent...")
    _, rewards_sarsa = train_sarsa()
    np.save("sarsa_rewards.npy", np.array(rewards_sarsa))
    print(f"Saved {len(rewards_sarsa)} episode rewards to sarsa_rewards.npy")


if __name__ == "__main__":
    main()
