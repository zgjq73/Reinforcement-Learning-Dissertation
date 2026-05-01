"""Train GRPO and PPO on Acrobot-v1."""

import numpy as np

from src.deep.policy_optim import train_grpo, train_ppo


SEED = 42
EPISODES = 500

if __name__ == "__main__":
    print("Training GRPO...")
    grpo_scores = train_grpo(episodes=EPISODES, seed=SEED)
    np.save("grpo_scores.npy", np.array(grpo_scores))
    print(f"Saved {len(grpo_scores)} episode scores to grpo_scores.npy")

    print("Training PPO...")
    ppo_scores = train_ppo(episodes=EPISODES, seed=SEED)
    np.save("ppo_scores.npy", np.array(ppo_scores))
    print(f"Saved {len(ppo_scores)} episode scores to ppo_scores.npy")
