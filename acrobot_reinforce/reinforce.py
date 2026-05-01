"""
REINFORCE family of policy-gradient methods on Acrobot-v1.

A single UniversalAgent class handles three variants by switching on `mode`:
    'REINFORCE'          - vanilla policy gradient
    'REINFORCE_Baseline' - policy gradient with value baseline
    'Actor_Critic'       - TD-based actor-critic (per-step updates)

Run directly with:  python acrobot_reinforce/reinforce.py
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym

from networks import PolicyNetwork, ValueNetwork


# ============================================================
# HYPERPARAMETERS
# ============================================================

# Seeds were varied between: 42, 123, 99, 10, 7
SEED = 42
EPISODES     = 500
LEARNING_RATE = 0.005
GAMMA        = 0.99
ENTROPY_COEF = 0.01
HIDDEN_SIZE  = 128


class UniversalAgent:
    def __init__(self, mode, state_dim, action_dim,
                 lr=LEARNING_RATE, gamma=GAMMA,
                 hidden_size=HIDDEN_SIZE, entropy_coef=ENTROPY_COEF):
        assert mode in ('REINFORCE', 'REINFORCE_Baseline', 'Actor_Critic')
        self.mode = mode
        self.gamma = gamma
        self.entropy_coef = entropy_coef

        # Every mode needs a policy network
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_size)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # Only Baseline and Actor-Critic need a value network
        self.value_net = None
        self.value_optimizer = None
        if mode in ('REINFORCE_Baseline', 'Actor_Critic'):
            self.value_net = ValueNetwork(state_dim, hidden_size)
            self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)

    def get_action(self, state):
        """Sample an action; return (action, log_prob, entropy)."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy_net(state_tensor)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), dist.entropy()

    def get_value(self, state):
        """Return V(state) if a value network exists, else 0."""
        if self.value_net is None:
            return torch.tensor(0.0)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        return self.value_net(state_tensor)

    # ---- End-of-episode update (REINFORCE and REINFORCE+Baseline) ----
    def update_episode(self, rewards, log_probs, values, entropies):
        # Discounted returns G_t for every time step
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns)

        # Normalise for numerical stability
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        policy_loss = []
        value_loss = []

        for t, log_prob in enumerate(log_probs):
            G_t = returns[t]
            entropy = entropies[t]

            if self.mode == 'REINFORCE_Baseline':
                V_s = values[t]
                advantage = G_t - V_s.item()
                policy_loss.append(-log_prob * advantage - self.entropy_coef * entropy)
                value_loss.append(F.mse_loss(V_s.squeeze(), G_t.float()))
            else:
                policy_loss.append(-log_prob * G_t - self.entropy_coef * entropy)

        self.policy_optimizer.zero_grad()
        sum(policy_loss).backward()
        self.policy_optimizer.step()

        if self.mode == 'REINFORCE_Baseline' and value_loss:
            self.value_optimizer.zero_grad()
            sum(value_loss).backward()
            self.value_optimizer.step()

    # ---- Per-step update (Actor-Critic) ----
    def update_step(self, state, next_state, reward, log_prob, done, entropy):
        curr_value = self.get_value(state)
        with torch.no_grad():
            next_value = self.get_value(next_state)

        target = reward + self.gamma * next_value * (1 - int(done))
        delta = target - curr_value

        critic_loss = F.mse_loss(curr_value, target.detach())
        actor_loss = -log_prob * delta.detach() - self.entropy_coef * entropy

        self.policy_optimizer.zero_grad()
        actor_loss.backward()
        self.policy_optimizer.step()

        self.value_optimizer.zero_grad()
        critic_loss.backward()
        self.value_optimizer.step()


def train_agent(mode, seed=SEED, episodes=EPISODES):
    """Train one of the three REINFORCE-family agents on Acrobot-v1."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make('Acrobot-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = UniversalAgent(mode, state_dim, action_dim)
    scores = []

    for episode in range(episodes):
        state, _ = env.reset(seed=seed + episode)
        score = 0
        done = False

        rewards, log_probs, values, entropies = [], [], [], []

        while not done:
            action, log_prob, entropy = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if mode == 'Actor_Critic':
                agent.update_step(state, next_state, reward, log_prob, done, entropy)
            else:
                rewards.append(reward)
                log_probs.append(log_prob)
                values.append(agent.get_value(state))
                entropies.append(entropy)

            score += reward
            state = next_state

        if mode != 'Actor_Critic':
            agent.update_episode(rewards, log_probs, values, entropies)

        scores.append(score)

    env.close()
    return scores


def main():
    for mode in ('REINFORCE', 'REINFORCE_Baseline', 'Actor_Critic'):
        print(f"Training {mode}...")
        scores = train_agent(mode)
        filename = f"{mode.lower()}_scores.npy"
        np.save(filename, np.array(scores))
        print(f"Saved {len(scores)} episode scores to {filename}")


if __name__ == "__main__":
    main()
