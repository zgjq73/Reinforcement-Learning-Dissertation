"""
GRPO and PPO agents on Acrobot-v1.

Both share the same policy-network architecture (defined here, with two
hidden layers and Tanh, to suit Acrobot's bounded continuous state space)
so the comparison is fair.

Key differences:
    GRPO - critic-free; advantages computed *relative to a group of episodes*;
           KL penalty against a slowly-updated frozen reference policy.
    PPO  - learned value function (critic); per-step advantages via GAE.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PolicyNetwork(nn.Module):
    """Tanh-MLP policy. Shared by both agents for fair comparison."""

    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return F.softmax(self.net(x), dim=-1)


# ============================================================
# GRPO
# ============================================================

class GRPOAgent:
    def __init__(self, state_dim, action_dim, lr=0.002, group_size=8,
                 entropy_coef=0.01, gamma=0.99, clip_eps=0.2,
                 kl_coef=0.01, update_epochs=4, max_grad_norm=0.5,
                 ref_update_freq=5):
        self.group_size = group_size
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.kl_coef = kl_coef
        self.update_epochs = update_epochs
        self.max_grad_norm = max_grad_norm
        self.ref_update_freq = ref_update_freq
        self.update_count = 0

        self.policy_net = PolicyNetwork(state_dim, action_dim).to(DEVICE)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # Frozen reference policy for the KL penalty
        self.ref_net = PolicyNetwork(state_dim, action_dim).to(DEVICE)
        self.ref_net.load_state_dict(self.policy_net.state_dict())
        for p in self.ref_net.parameters():
            p.requires_grad = False

    def get_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs = self.policy_net(state_tensor)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action).cpu()

    def update(self, group_states, group_actions, group_log_probs, group_rewards):
        """Update from a group of completed episodes."""
        n_episodes = len(group_rewards)
        if n_episodes < 2:
            return

        # Group-relative advantages: each episode gets one number,
        # how much better its total return was vs. the group mean.
        episode_returns = np.array([sum(r) for r in group_rewards])
        mean_ret = episode_returns.mean()
        std_ret = episode_returns.std() + 1e-8

        advantages_list = []
        for i in range(n_episodes):
            adv = (episode_returns[i] - mean_ret) / std_ret
            advantages_list.extend([adv] * len(group_rewards[i]))

        advantages_tensor = torch.tensor(advantages_list, dtype=torch.float32).to(DEVICE)
        flat_states = torch.FloatTensor(np.concatenate(group_states)).to(DEVICE)
        flat_actions = torch.tensor(np.concatenate(group_actions), dtype=torch.long).to(DEVICE)
        flat_old_log_probs = torch.cat(
            [lp for sublist in group_log_probs for lp in sublist]
        ).to(DEVICE)

        with torch.no_grad():
            ref_probs = self.ref_net(flat_states)

        for _ in range(self.update_epochs):
            probs = self.policy_net(flat_states)
            dist = torch.distributions.Categorical(probs)
            new_log_probs = dist.log_prob(flat_actions)
            entropy = dist.entropy()

            ratios = torch.exp(new_log_probs - flat_old_log_probs)
            surr1 = ratios * advantages_tensor
            surr2 = torch.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages_tensor
            surrogate_loss = torch.min(surr1, surr2).mean()

            if self.kl_coef > 0:
                kl_div = F.kl_div(ref_probs.log(), probs, reduction='batchmean', log_target=False)
            else:
                kl_div = torch.tensor(0.0).to(DEVICE)

            entropy_bonus = entropy.mean()
            loss = -(surrogate_loss - self.kl_coef * kl_div + self.entropy_coef * entropy_bonus)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
            self.optimizer.step()

        # Periodically refresh the reference policy
        self.update_count += 1
        if self.update_count % self.ref_update_freq == 0:
            self.ref_net.load_state_dict(self.policy_net.state_dict())


# ============================================================
# PPO
# ============================================================

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=0.002, gamma=0.99,
                 gae_lambda=0.95, clip_eps=0.2, entropy_coef=0.01,
                 value_coef=0.5, update_epochs=4, max_grad_norm=0.5,
                 minibatch_size=64):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.update_epochs = update_epochs
        self.max_grad_norm = max_grad_norm
        self.minibatch_size = minibatch_size

        self.policy_net = PolicyNetwork(state_dim, action_dim).to(DEVICE)
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1),
        ).to(DEVICE)

        self.optimizer = optim.Adam(
            list(self.policy_net.parameters()) + list(self.value_net.parameters()),
            lr=lr,
        )

    def get_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs = self.policy_net(state_tensor)
            value = self.value_net(state_tensor)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action).cpu(), value.item()

    def compute_gae(self, rewards, values, dones):
        """Generalised Advantage Estimation."""
        advantages = []
        gae = 0.0
        next_value = 0.0
        for t in reversed(range(len(rewards))):
            if dones[t]:
                next_value = 0.0
                gae = 0.0
            delta = rewards[t] + self.gamma * next_value - values[t]
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages.insert(0, gae)
            next_value = values[t]
        returns = [adv + val for adv, val in zip(advantages, values)]
        return advantages, returns

    def update(self, states, actions, log_probs, rewards, values, dones):
        advantages, returns = self.compute_gae(rewards, values, dones)

        states_t = torch.FloatTensor(np.array(states)).to(DEVICE)
        actions_t = torch.tensor(actions, dtype=torch.long).to(DEVICE)
        old_log_probs_t = torch.cat(log_probs).to(DEVICE)
        advantages_t = torch.tensor(advantages, dtype=torch.float32).to(DEVICE)
        returns_t = torch.tensor(returns, dtype=torch.float32).to(DEVICE)

        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        all_params = list(self.policy_net.parameters()) + list(self.value_net.parameters())

        for _ in range(self.update_epochs):
            indices = np.random.permutation(len(states))
            for start in range(0, len(states), self.minibatch_size):
                mb = indices[start:start + self.minibatch_size]

                probs = self.policy_net(states_t[mb])
                dist = torch.distributions.Categorical(probs)
                new_log = dist.log_prob(actions_t[mb])
                entropy = dist.entropy()
                values_pred = self.value_net(states_t[mb]).squeeze(-1)

                ratios = torch.exp(new_log - old_log_probs_t[mb])
                surr1 = ratios * advantages_t[mb]
                surr2 = torch.clamp(ratios, 1 - self.clip_eps, 1 + self.clip_eps) * advantages_t[mb]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values_pred, returns_t[mb])

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(all_params, self.max_grad_norm)
                self.optimizer.step()


# ============================================================
# Training entry points
# ============================================================

def train_grpo(config=None, episodes=500, seed=42):
    """Train GRPO and return the per-episode score list."""
    config = config or {}
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make('Acrobot-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = GRPOAgent(
        state_dim, action_dim,
        lr=config.get('lr', 0.002),
        group_size=config.get('group_size', 8),
        entropy_coef=config.get('entropy_coef', 0.01),
        gamma=config.get('gamma', 0.99),
        clip_eps=config.get('clip_eps', 0.2),
        kl_coef=config.get('kl_coef', 0.01),
        update_epochs=config.get('update_epochs', 4),
        max_grad_norm=config.get('max_grad_norm', 0.5),
    )

    all_scores = []

    while len(all_scores) < episodes:
        group_states, group_actions, group_rewards, group_log_probs = [], [], [], []

        for _ in range(agent.group_size):
            if len(all_scores) >= episodes:
                break

            state, _ = env.reset(seed=seed + len(all_scores))
            done = False
            ep_states, ep_actions, ep_rewards, ep_log_probs = [], [], [], []
            score = 0

            while not done:
                action, log_prob = agent.get_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                ep_states.append(state)
                ep_actions.append(action)
                ep_rewards.append(reward)
                ep_log_probs.append(log_prob)
                score += reward
                state = next_state

            group_states.append(ep_states)
            group_actions.append(ep_actions)
            group_rewards.append(ep_rewards)
            group_log_probs.append(ep_log_probs)
            all_scores.append(score)

        if len(group_rewards) > 1:
            agent.update(group_states, group_actions, group_log_probs, group_rewards)

    env.close()
    return all_scores


def train_ppo(config=None, episodes=500, seed=42, collect_steps=2048):
    """Train PPO and return the per-episode score list."""
    config = config or {}
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make('Acrobot-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = PPOAgent(
        state_dim, action_dim,
        lr=config.get('lr', 0.002),
        gamma=config.get('gamma', 0.99),
        clip_eps=config.get('clip_eps', 0.2),
        entropy_coef=config.get('entropy_coef', 0.01),
        update_epochs=config.get('update_epochs', 4),
        max_grad_norm=config.get('max_grad_norm', 0.5),
    )

    all_scores = []
    ep_count = 0

    while ep_count < episodes:
        states, actions, log_probs, rewards, values, dones = [], [], [], [], [], []
        steps_collected = 0

        while steps_collected < collect_steps and ep_count < episodes:
            state, _ = env.reset(seed=seed + ep_count)
            done = False
            score = 0

            while not done:
                action, log_prob, value = agent.get_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                states.append(state)
                actions.append(action)
                log_probs.append(log_prob)
                rewards.append(reward)
                values.append(value)
                dones.append(done)
                score += reward
                state = next_state
                steps_collected += 1

            ep_count += 1
            all_scores.append(score)

        if len(states) > 0:
            agent.update(states, actions, log_probs, rewards, values, dones)

    env.close()
    return all_scores
