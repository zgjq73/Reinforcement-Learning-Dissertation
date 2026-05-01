"""
GRPO and PPO on Acrobot-v1.

Both share the same policy-network architecture (defined here, with two
hidden layers and Tanh, to suit Acrobot's bounded continuous state space)
so the comparison is fair.

Key differences:
    GRPO - critic-free; advantages computed *relative to a group of episodes*;
           KL penalty against a slowly-updated frozen reference policy.
    PPO  - learned value function (critic); per-step advantages via GAE.

Run directly with:  python acrobot_grpo_ppo/grpo_ppo.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym


# ============================================================
# HYPERPARAMETERS
# ============================================================

# Seeds used in the dissertation: 42, 123, 99, 10, 7
SEED = 42
EPISODES = 500
DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================
# POLICY NETWORK (shared by GRPO and PPO)
# ============================================================
# Feedforward network mapping state -> action probabilities pi_theta(a|s).
# 6 inputs (Acrobot state), 3 outputs (left, zero, right torque),
# two hidden layers of 64 units with tanh activations.

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
# GRPO agent and baseline hyperparams. Note that the the hyperparam values
# were changed one at a time for the ablation
# ============================================================
# Ablation configurations run for the dissertation:
    #   group_size:    4, 8 (baseline), 16, 32
    #   clip_eps:      0.1, 0.2 (baseline), 0.3
    #   kl_coef:       0.0, 0.01 (baseline), 0.04
    #   entropy_coef:  0.0, 0.01 (baseline), 0.05
# all ablations were ran across all five seeds.

class GRPOAgent:
    def __init__(self, state_dim, action_dim, lr=0.002, group_size=8,
                 entropy_coef=0.01, gamma=0.99, clip_eps=0.2,
                 kl_coef=0.01, update_epochs=4, max_grad_norm=0.5,
                 ref_update_freq=5):
        self.group_size = group_size            # G in the GRPO objective
        self.entropy_coef = entropy_coef        # alpha_H the entropy coefficient
        self.gamma = gamma                      # discount factor
        self.clip_eps = clip_eps                # epsilon in the clip term
        self.kl_coef = kl_coef                  # beta on the KL penalty
        self.update_epochs = update_epochs      # K passes over each group
        self.max_grad_norm = max_grad_norm
        self.ref_update_freq = ref_update_freq  # refresh pi_ref every N updates
        self.update_count = 0
        
        # Uses adam optimiser as defined in appendix
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
            
        # Step 2: K epochs of gradient updates on the GRPO objective defined in chapter 5. Note entropy added directly to entropy term.
        for _ in range(self.update_epochs):
            probs = self.policy_net(flat_states)
            dist = torch.distributions.Categorical(probs)
            new_log_probs = dist.log_prob(flat_actions)
            entropy = dist.entropy()

            # clipped surogate using epislon value (baseline 0.2)
            
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
    # PPO uses the same clipped surrogate as GRPO but computes per-step
    # advantages from a learned value function V_w(s) using GAE,
    # rather than a single per-episode group-relative advantage.
 
    def __init__(self, state_dim, action_dim, lr=0.002, gamma=0.99,
                 gae_lambda=0.95, clip_eps=0.2, entropy_coef=0.01,
                 value_coef=0.5, update_epochs=4, max_grad_norm=0.5,
                 minibatch_size=64):
        self.gamma = gamma
        self.gae_lambda = gae_lambda            # lambda in GAE
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef            # weight on the critic loss
        self.update_epochs = update_epochs
        self.max_grad_norm = max_grad_norm
        self.minibatch_size = minibatch_size
 
        # Actor: same architecture as GRPO for fair comparison.
        self.policy_net = PolicyNetwork(state_dim, action_dim).to(DEVICE)
 
        # Critic V_w(s): an extra network mapping state -> scalar value.
        # Doubles the parameter count vs GRPO but provides per-step
        # advantage estimates instead of one number per episode.
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1),
        ).to(DEVICE)
 
        # Single optimiser updates both networks jointly.
        self.optimizer = optim.Adam(
            list(self.policy_net.parameters()) + list(self.value_net.parameters()),
            lr=lr,
        )
 
    def get_action(self, state):
        # Forward both networks: the actor gives action probabilities,
        # the critic gives a value estimate V_w(s) used later in GAE.
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs = self.policy_net(state_tensor)
            value = self.value_net(state_tensor)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action).cpu(), value.item()
 
    def compute_gae(self, rewards, values, dones):
        """
        Generalised Advantage Estimation recursive rule defined at end of chapter 4
 
        At each step computes the TD residual
            delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        then forms the advantage as a discounted sum of future deltas:
            A_t = delta_t + (gamma * lambda) * delta_{t+1} + ...
 
        lambda=0.95 used as standard.
        """
        advantages = []
        gae = 0.0
        next_value = 0.0
        # Iterate backwards so each step's advantage can use the next.
        for t in reversed(range(len(rewards))):
            # When an episode ends, reset the bootstrap value and GAE
            # accumulator so that advantages don't leak across episodes.
            if dones[t]:
                next_value = 0.0
                gae = 0.0
            delta = rewards[t] + self.gamma * next_value - values[t]
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages.insert(0, gae)
            next_value = values[t]
        # Returns = advantage + value, used as the regression target
        # for the critic loss.
        returns = [adv + val for adv, val in zip(advantages, values)]
        return advantages, returns
 
    def update(self, states, actions, log_probs, rewards, values, dones):
        # Compute per-step advantages and value targets via GAE defined in chapter 4.
        advantages, returns = self.compute_gae(rewards, values, dones)
 
        states_t = torch.FloatTensor(np.array(states)).to(DEVICE)
        actions_t = torch.tensor(actions, dtype=torch.long).to(DEVICE)
        old_log_probs_t = torch.cat(log_probs).to(DEVICE)
        advantages_t = torch.tensor(advantages, dtype=torch.float32).to(DEVICE)
        returns_t = torch.tensor(returns, dtype=torch.float32).to(DEVICE)
 
        # Standardise advantages for variance reduction — a common PPO trick
        # that does NOT change the optimal policy but stabilises gradients.
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
 
        all_params = list(self.policy_net.parameters()) + list(self.value_net.parameters())
 
        # K epochs over the data, with random minibatches per epoch.
        for _ in range(self.update_epochs):
            indices = np.random.permutation(len(states))
            for start in range(0, len(states), self.minibatch_size):
                mb = indices[start:start + self.minibatch_size]
 
                # Re-evaluate the current policy and value function on this minibatch.
                probs = self.policy_net(states_t[mb])
                dist = torch.distributions.Categorical(probs)
                new_log = dist.log_prob(actions_t[mb])
                entropy = dist.entropy()
                values_pred = self.value_net(states_t[mb]).squeeze(-1)
 
                # Importance ratio R = pi_theta / pi_theta_old, same as GRPO.
                ratios = torch.exp(new_log - old_log_probs_t[mb])
                surr1 = ratios * advantages_t[mb]
                surr2 = torch.clamp(ratios, 1 - self.clip_eps, 1 + self.clip_eps) * advantages_t[mb]
                # Negative because we minimise the loss but maximise the surrogate.
                policy_loss = -torch.min(surr1, surr2).mean()
 
                # Critic loss: MSE between predicted V_w(s) and the GAE return target.
                value_loss = F.mse_loss(values_pred, returns_t[mb])
 
                # Combined PPO loss: actor loss + critic loss - entropy bonus.
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()
 
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(all_params, self.max_grad_norm)
                self.optimizer.step()
 
 
# ============================================================
# Training entry points
# ============================================================
 
def train_grpo(episodes=EPISODES, seed=SEED):
    """Train GRPO and return per-episode total rewards (used for plotting)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
 
    env = gym.make('Acrobot-v1')
    state_dim = env.observation_space.shape[0]   # 6
    action_dim = env.action_space.n              # 3
    agent = GRPOAgent(state_dim, action_dim)
    all_scores = []
 
    # Outer loop: keep collecting groups until we hit the episode budget.
    while len(all_scores) < episodes:
        # One group = G episodes rolled out under the same policy.
        group_states, group_actions, group_rewards, group_log_probs = [], [], [], []
 
        for _ in range(agent.group_size):
            if len(all_scores) >= episodes:
                break
 
            # Roll out a single episode under the current pi_theta.
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
 
        # Once the group is full, do one GRPO update across all G episodes.
        if len(group_rewards) > 1:
            agent.update(group_states, group_actions, group_log_probs, group_rewards)
 
    env.close()
    return all_scores
 
 
def train_ppo(episodes=EPISODES, seed=SEED, collect_steps=2048):
    """Train PPO and return per-episode total rewards."""
    torch.manual_seed(seed)
    np.random.seed(seed)
 
    env = gym.make('Acrobot-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = PPOAgent(state_dim, action_dim)
    all_scores = []
    ep_count = 0
 
    # PPO collects a fixed number of *steps* per update (not episodes),
    # which may span partial or multiple episodes.
    while ep_count < episodes:
        states, actions, log_probs, rewards, values, dones = [], [], [], [], [], []
        steps_collected = 0
 
        # Keep rolling out episodes until we have ~collect_steps transitions.
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
 
        # One PPO update per collected batch.
        if len(states) > 0:
            agent.update(states, actions, log_probs, rewards, values, dones)
 
    env.close()
    return all_scores
 
 
def main():
    print("Training GRPO...")
    grpo_scores = train_grpo()
    np.save("grpo_scores.npy", np.array(grpo_scores))
    print(f"Saved {len(grpo_scores)} episode scores to grpo_scores.npy")
 
    print("Training PPO...")
    ppo_scores = train_ppo()
    np.save("ppo_scores.npy", np.array(ppo_scores))
    print(f"Saved {len(ppo_scores)} episode scores to ppo_scores.npy")
 
 
if __name__ == "__main__":
    main()
