"""Train Q-Learning and SARSA agents and save their reward histories."""

import random
import numpy as np

from src.tabular.td_methods import train_q_learning, train_sarsa


SEED = 42

if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)

    print("Training Q-Learning agent...")
    _, rewards_ql = train_q_learning(episodes=1200)
    np.save("ql_rewards.npy", np.array(rewards_ql))
    print(f"Saved {len(rewards_ql)} episode rewards to ql_rewards.npy")

    # Re-seed before SARSA so it sees the same noise stream as Q-Learning did
    random.seed(SEED)
    np.random.seed(SEED)

    print("Training SARSA agent...")
    _, rewards_sarsa = train_sarsa(episodes=1200)
    np.save("sarsa_rewards.npy", np.array(rewards_sarsa))
    print(f"Saved {len(rewards_sarsa)} episode rewards to sarsa_rewards.npy")
