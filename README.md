# Reinforcement Learning Algorithms — Undergraduate Dissertation

A from-scratch implementation of seven reinforcement learning algorithms, written as the practical companion to my undergraduate mathematics dissertation.

## Algorithms

| Family | Algorithm | Environment | File |
|---|---|---|---|
| Tabular | Monte Carlo (First-Visit) | 5×5 Gridworld | `src/tabular/monte_carlo.py` |
| Tabular | Q-Learning | 5×5 Gridworld | `src/tabular/td_methods.py` |
| Tabular | SARSA | 5×5 Gridworld | `src/tabular/td_methods.py` |
| Deep | REINFORCE | Acrobot-v1 | `src/deep/reinforce.py` |
| Deep | REINFORCE with Baseline | Acrobot-v1 | `src/deep/reinforce.py` |
| Deep | Actor-Critic | Acrobot-v1 | `src/deep/reinforce.py` |
| Deep | GRPO | Acrobot-v1 | `src/deep/policy_optim.py` |
| Deep | PPO | Acrobot-v1 | `src/deep/policy_optim.py` |

The tabular methods run on a small gridworld with stochastic dynamics ("wind") and lava pits, designed to expose the cautious-vs-aggressive distinction between SARSA and Q-Learning. The deep methods are evaluated on Acrobot-v1, a continuous-state control task from Gymnasium.

## Repository Structure

```
rl-dissertation/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── src/
│   ├── tabular/
│   │   ├── environment.py     ← MarioEnvironment (gridworld)
│   │   ├── policies.py        ← greedy & ε-greedy action selection
│   │   ├── monte_carlo.py     ← First-Visit MC training
│   │   └── td_methods.py      ← Q-Learning & SARSA training
│   └── deep/
│       ├── networks.py        ← Policy & Value networks (PyTorch)
│       ├── reinforce.py       ← REINFORCE family
│       └── policy_optim.py    ← GRPO & PPO
└── experiments/
    ├── train_mc.py            ← runnable training entry points
    ├── train_td.py
    ├── train_reinforce.py
    └── train_grpo_ppo.py
```

## Quick Start

```bash
# 1. Clone
git clone https://github.com/<your-username>/rl-dissertation.git
cd rl-dissertation

# 2. Install dependencies (Python 3.10+)
pip install -r requirements.txt

# 3. Run an experiment
python -m experiments.train_mc
python -m experiments.train_td
python -m experiments.train_reinforce
python -m experiments.train_grpo_ppo
```

Each training script saves its reward history to a `.npy` file in the project root, ready for plotting.

## Notes

- Random seeds are set in every training script for reproducibility.
- Hyperparameters live at the top of each experiment file as named constants.
- The tabular environment (`MarioEnvironment`) and helper policies are defined once in `src/tabular/` and imported where needed — no duplication between the Monte Carlo and TD scripts.

## License

MIT — see `LICENSE`.
