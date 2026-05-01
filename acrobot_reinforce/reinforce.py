"""
REINFORCE family of policy-gradient methods on Acrobot-v1.

A single UniversalAgent class handles three variants by switching on `mode`:
    'REINFORCE'          - vanilla policy gradient (Eq: reinforce-update)
    'REINFORCE_Baseline' - policy gradient with learned value baseline
    'Actor_Critic'       - TD-based actor-critic (per-step updates)

All three optimise the same policy gradient theorem objective; they
differ only in how they estimate the advantage that scales each
log-probability gradient.

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

# Seeds used in the dissertation: 42, 123, 99, 10, 7
# Each seed was run individually by changing this value and rerunning.
SEED          = 42
EPISODES      = 500
LEARNING_RATE = 0.005
GAMMA         = 0.99    # discount factor
ENTROPY_COEF  = 0.01    # alpha_H — same exploration bonus as in GRPO
HIDDEN_SIZE   = 128


class UniversalAgent:
    # One class, three algorithms — what changes is the advantage estimator:
    # mathematical foundations of this code defined in chapter 4
    #   REINFORCE          : A_hat = G_t                 (full return)
    #   REINFORCE_Baseline : A_hat = G_t - V_w(s_t)      (return - baseline)
    #   Actor_Critic       : A_hat = r + gamma*V(s')-V(s) (TD residual)

    def __init__(self, mode, state_dim, action_dim,
                 lr=LEARNING_RATE, gamma=GAMMA,
                 hidden_size=HIDDEN_SIZE, entropy_coef=ENTROPY_COEF):
        assert mode in ('REINFORCE', 'REINFORCE_Baseline', 'Actor_Critic')
        self.mode = mode
        self.gamma = gamma
        self.entropy_coef = entropy_coef

        # Actor pi_theta — every variant has one. Outputs action probabilities.
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_size)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # Critic V_w(s) — only used by Baseline and Actor-Critic.
        # Vanilla REINFORCE has no value network.
        self.value_net = None
        self.value_optimizer = None
        if mode in ('REINFORCE_Baseline', 'Actor_Critic'):
            self.value_net = ValueNetwork(state_dim, hidden_size)
            self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)

    def get_action(self, state):
        """Sample a~pi_theta(.|s) and return the action with its log-prob and entropy."""
        # Forward pass through the policy network gives action probabilities.
        # Note: gradient IS tracked here (no torch.no_grad()) because we need
        # the log-prob to be differentiable for the policy gradient update.
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy_net(state_tensor)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        # log_prob is log pi_theta(a|s) — the term that gets multiplied by
        # the advantage in the policy gradient update.
        # entropy is H(pi_theta(.|s)) — used for the exploration bonus.
        return action.item(), dist.log_prob(action), dist.entropy()

    def get_value(self, state):
        """Return V_w(state) if a critic exists, else 0."""
        if self.value_net is None:
            return torch.tensor(0.0)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        return self.value_net(state_tensor)

    # ============================================================
    # End-of-episode update (REINFORCE and REINFORCE+Baseline)
    # ============================================================
    # Both methods only update once per episode because they need the
    # full return G_t to compute their advantage estimate.

    def update_episode(self, rewards, log_probs, values, entropies):
        # ------------------------------------------------------------
        # Step 1: compute discounted returns G_t for every timestep,
        # working backwards so each G_t reuses the next:
        #     G_t = r_t + gamma * G_{t+1}
        # ------------------------------------------------------------
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns)

        # Standardise returns to reduce gradient variance.
        # This is a common trick — it changes the magnitude of updates
        # but not the gradient direction, so the optimal policy is unchanged.
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        policy_loss = []
        value_loss = []

        # ------------------------------------------------------------
        # Step 2: build per-timestep losses. Note for the code implementation
        # the objective function is reverse and minimised as if it was a loss function
        # so the *loss* (which we minimise) is -A_hat * log pi_theta.
        # ------------------------------------------------------------
        for t, log_prob in enumerate(log_probs):
            G_t = returns[t]
            entropy = entropies[t]

            if self.mode == 'REINFORCE_Baseline':
                # Subtract the learned baseline V_w(s_t) to reduce variance.
                # The expected gradient is unchanged because E[V(s) * grad log pi] = 0.
                V_s = values[t]
                advantage = G_t - V_s.item()  # detach: don't backprop through V
                policy_loss.append(-log_prob * advantage - self.entropy_coef * entropy)
                # Critic regresses V_w(s_t) toward the observed return G_t.
                value_loss.append(F.mse_loss(V_s.squeeze(), G_t.float()))
            else:
                # Vanilla REINFORCE: advantage IS the (standardised) return.
                policy_loss.append(-log_prob * G_t - self.entropy_coef * entropy)

        # ------------------------------------------------------------
        # Step 3: backprop policy 
        # ------------------------------------------------------------
        self.policy_optimizer.zero_grad()
        sum(policy_loss).backward()
        self.policy_optimizer.step()

        if self.mode == 'REINFORCE_Baseline' and value_loss:
            self.value_optimizer.zero_grad()
            sum(value_loss).backward()
            self.value_optimizer.step()

    # ============================================================
    # Per-step update (Actor-Critic)
    # ============================================================
    # Actor-Critic doesn't wait for the episode to end — it uses the
    # 1-step TD residual as the advantage and updates after every action.

    def update_step(self, state, next_state, reward, log_prob, done, entropy):
        curr_value = self.get_value(state)
        # No gradient through V(s'): the bootstrap target is treated as a constant.
        with torch.no_grad():
            next_value = self.get_value(next_state)

        # TD target: r + gamma * V(s'), zeroed at terminal states because
        # no future reward exists past the end of an episode.
        target = reward + self.gamma * next_value * (1 - int(done))
        # TD residual delta = target - V(s) — used as the advantage estimate.
        delta = target - curr_value

        # Critic regresses V_w(s) toward the TD target.
        critic_loss = F.mse_loss(curr_value, target.detach())
        # Actor uses delta (detached: gradient flows only through log_prob).
        actor_loss = -log_prob * delta.detach() - self.entropy_coef * entropy

        # Backprop actor and critic separately — they don't share parameters.
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
    state_dim = env.observation_space.shape[0]   # 6
    action_dim = env.action_space.n              # 3

    agent = UniversalAgent(mode, state_dim, action_dim)
    scores = []

    for episode in range(episodes):
        state, _ = env.reset(seed=seed + episode)
        score = 0
        done = False

        # Buffers for episodic methods (REINFORCE / Baseline).
        # Actor-Critic ignores these because it updates per-step.
        rewards, log_probs, values, entropies = [], [], [], []

        while not done:
            action, log_prob, entropy = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if mode == 'Actor_Critic':
                # Online update: one gradient step per environment step.
                agent.update_step(state, next_state, reward, log_prob, done, entropy)
            else:
                # Store transitions for the end-of-episode update.
                rewards.append(reward)
                log_probs.append(log_prob)
                values.append(agent.get_value(state))
                entropies.append(entropy)

            score += reward
            state = next_state

        # REINFORCE-style methods do their gradient update after the episode ends.
        if mode != 'Actor_Critic':
            agent.update_episode(rewards, log_probs, values, entropies)

        scores.append(score)

    env.close()
    return scores


def main():
    # Train all three variants in sequence with the current SEED.
    # To produce the 5-seed averages used in the dissertation, change SEED
    # at the top of this file to each of {42, 123, 99, 10, 7} and rerun.
    for mode in ('REINFORCE', 'REINFORCE_Baseline', 'Actor_Critic'):
        print(f"Training {mode}...")
        scores = train_agent(mode)
        filename = f"{mode.lower()}_scores.npy"
        np.save(filename, np.array(scores))
        print(f"Saved {len(scores)} episode scores to {filename}")


if __name__ == "__main__":
    main()
