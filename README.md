# Reinforcement Learning Code — Undergraduate Dissertation

Dissertation implementation code. This repository includes the code for all of the experiments which were ran within my dissertation. We also included the results of an additional experiment which was provided in the appendix.

## Reposity Contents
The layout is the same structure as in the project with the gridworld showing implementations from chapter 3, acrobot reinforce showing implementations from chapter 4 and acrobot grpo_ppo showing implementations from chapter 6.

```
rl-dissertation/
├── README.md
│
├── gridworld/              ← Tabular methods on a 5x5 grid
│   ├── environment.py      ← The gridworld (lava, wind, goal)
│   ├── policies.py         ← Greedy and epsilon-greedy action selection
│   ├── monte_carlo.py      ← First-Visit Monte Carlo
│   └── td_methods.py       ← Q-Learning and SARSA
│
├── acrobot_reinforce/      ← Policy-gradient methods on Acrobot-v1
│   ├── networks.py         ← Policy and Value neural networks
│   └── reinforce.py        ← REINFORCE, REINFORCE+Baseline, Actor-Critic
│
└── acrobot_grpo_ppo/       ← GRPO and PPO on Acrobot-v1
    └── grpo_ppo.py         ← Both agents and their training loops
```

## Algorithms

| Folder | Algorithm | Environment |
|---|---|---|
| `gridworld/` | Monte Carlo (First-Visit) | 5×5 Gridworld |
| `gridworld/` | Q-Learning | 5×5 Gridworld |
| `gridworld/` | SARSA | 5×5 Gridworld |
| `acrobot_reinforce/` | REINFORCE | Acrobot-v1 |
| `acrobot_reinforce/` | REINFORCE with Baseline | Acrobot-v1 |
| `acrobot_reinforce/` | Actor-Critic | Acrobot-v1 |
| `acrobot_grpo_ppo/` | GRPO | Acrobot-v1 |
| `acrobot_grpo_ppo/` | PPO | Acrobot-v1 |

## The Three Experiments

### `gridworld/` — Tabular Methods - Chapter 3

A 5×5 grid where the agent must reach the goal while avoiding lava pits. A 20% "wind" chance redirects the agent's action to a random direction, making the environment stochastic. Used to compare three classic tabular methods, in particular to surface the cautious-vs-aggressive distinction between SARSA and Q-Learning.

### `acrobot_reinforce/` — REINFORCE Family - Chapter 4

The Acrobot-v1 environment from Gymnasium: a two-link pendulum that must swing its tip above a target height. State is six-dimensional and continuous, actions are three discrete torques. A single `UniversalAgent` class handles all three policy-gradient variants by switching on a `mode` argument.

### `acrobot_grpo_ppo/` — GRPO and PPO - Chapter 6

Same Acrobot environment, used for the GRPO ablation study in the final dissertation chapter. GRPO is critic-free with group-relative advantages; PPO uses a learned value function and GAE.

## Plots and Data logging

Training metrics were logged live throughout each run using Weights & Biases , which allowed experiments to be monitored as they progressed and any issues to be identified early. All figures presented in this chapter were exported directly from the W&B project dashboard.

## Appendix - Initial Preliminary experiment - motivation for Acrobot Pivot
### RLVR Linear congruences

The logged experiment for the finetuning test experiment was recorded on Wandb website.
Here is the report for the prelim experiment if interested. 
Results show agent was unable to reason. Will notice the reward hacking by looking at the table.

By looking at the table you can observe the recorded prompts and responses
https://wandb.ai/oscarztimms-durham-university/grpo-linear-congruence/reports/Prelim-experiments--VmlldzoxNjczNzMwOA


### AI Declaration

Claude 4.5 used to, refactor, debug and improve the efficiency of code. Used to structure this repository and improve clarity by adding in comments.
