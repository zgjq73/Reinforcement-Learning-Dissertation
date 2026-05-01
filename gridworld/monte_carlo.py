"""
First-Visit Monte Carlo control on the gridworld.

Run directly with:  python gridworld/monte_carlo.py
"""

import random
import numpy as np

from environment import MarioEnvironment
from policies import epsilon_greedy


# ============================================================
# HYPERPARAMETERS
# ============================================================

SEED        = 42
EPISODES    = 10000
ALPHA       = 0.05
GAMMA       = 0.95
MAX_EPSILON = 1.0
MIN_EPSILON = 0.01
DECAY_RATE  = 0.999


def run_episode(env, table, epsilon):
    """Run one full episode using epsilon-greedy and return the trajectory."""
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


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    env = MarioEnvironment()
    Qtable = np.zeros((env.total_states, env.action_dimensions))
    epsilon = MAX_EPSILON
    episode_rewards = []

    print("Training Monte Carlo agent...")

    for ep in range(1, EPISODES + 1):
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
            G = GAMMA * G + reward

            visited_before = any(
                trajectory[i][0] == state and trajectory[i][1] == action
                for i in range(t)
            )
            if not visited_before:
                Qtable[state, action] += ALPHA * (G - Qtable[state, action])

        # Step 3: decay exploration rate
        epsilon = max(MIN_EPSILON, epsilon * DECAY_RATE)

    np.save("mc_rewards.npy", np.array(episode_rewards))
    print(f"Saved {len(episode_rewards)} episode rewards to mc_rewards.npy")


if __name__ == "__main__":
    main()
