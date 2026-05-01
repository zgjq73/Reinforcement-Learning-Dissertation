"""Train REINFORCE, REINFORCE+Baseline, and Actor-Critic on Acrobot-v1."""

import numpy as np

from src.deep.reinforce import train_agent


SEED = 42
EPISODES = 500

if __name__ == "__main__":
    for mode in ('REINFORCE', 'REINFORCE_Baseline', 'Actor_Critic'):
        print(f"Training {mode}...")
        scores = train_agent(mode, seed=SEED, episodes=EPISODES)
        filename = f"{mode.lower()}_scores.npy"
        np.save(filename, np.array(scores))
        print(f"Saved {len(scores)} episode scores to {filename}")
