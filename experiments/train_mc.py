"""Train the First-Visit Monte Carlo agent and save the reward history."""

import random
import numpy as np

from src.tabular.monte_carlo import train_monte_carlo


SEED = 42

if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)

    print("Training Monte Carlo agent...")
    Qtable, rewards = train_monte_carlo(
        episodes=10000,
        alpha=0.05,
        gamma=0.95,
        max_epsilon=1.0,
        min_epsilon=0.01,
        decay_rate=0.999,
    )

    np.save("mc_rewards.npy", np.array(rewards))
    print(f"Saved {len(rewards)} episode rewards to mc_rewards.npy")
