# CleanMBRL

Single-file implementations of model-based reinforcement learning algorithms.

## What's in here

This repo implements **7 MBRL algorithms** across **12 environments**, from discrete control to pixel-based DMControl tasks. Every algorithm is self-contained — no 2000-line monoliths, no hidden dependencies. Just clean PyTorch.

### Algorithms

| Algorithm | Approach | Key Idea |
|-----------|----------|----------|
| **DreamerV3** | RSSM world model | Learn a compact latent world model from pixels, plan in latent space. The gold standard for pixel-based MBRL. |
| **MuZero** | MCTS + learned model | No access to environment dynamics. Learns a predictive model and plans with Monte Carlo Tree Search. |
| **MuZero + EfficientZero** | MCTS + consistency fix | Fixes MuZero's reward/value representation mismatch with a consistency loss. |
| **MBPO** | Ensemble dynamics | Model-based policy optimization. Train an ensemble of dynamics models, roll out virtually, optimize policy. |
| **PETS** | Probabilistic ensemble | Uses Bayesian neural network ensembles for uncertainty-aware model rollout and planning. |
| **TD-MPC2** | Latent planning | TD learning with massively parallel policy optimization via cross-entropy method in latent space. |
| **JEPA (N-body)** | Self-supervised world model | Predicts latent representations of future states. No pixel reconstruction — just consistency in the latent space. Tested on N-body physics. |

### Environments

- **CartPole-v1** — discrete, low-dimensional (MuZero, PETS)
- **Pendulum-v1** — continuous, low-dimensional (PETS, TD-MPC2)
- **HalfCheetah-v5, Hopper-v5** — continuous, MuJoCo (MBPO, PETS)
- **DMControl (pixel-based)** — images as observations (DreamerV3)
- **N-body physics** — custom physics simulation for JEPA

## Quick start

```bash
# Install (requires Python 3.12+)
pip install -e ".[dev]"

# Or with uv (faster)
uv sync --all-extras
```

### Run an algorithm

Every script uses `tyro` for CLI args — just run it and tweak what you need:

```bash
# MuZero on CartPole (solves in ~40k steps)
python muzero_cartpole.py

# MBPO on HalfCheetah
python mbpo_halfcheetah.py

# TD-MPC2 on Pendulum
python tdmpc2_pendulum.py

# PETS on CartPole
python pets_cartpole.py

# DreamerV3 on DMControl (walker_walk)
python -m dreamerv3.train

# JEPA on N-body physics
python jepa_nbody/train.py
```

All scripts support `--track` for Weights & Biases logging and accept any dataclass field as a CLI override:

```bash
python muzero_cartpole.py --total_steps 80000 --seed 42 --track
```

### Visualize results

```bash
# Parse logs and generate comparison plots
python plot_results.py

# Watch a trained agent
python viz.py

# Record a trained agent as GIF/MP4
python record_agent.py
```

## Running on Apple Silicon

This is developed on an M5 Pro chip. Everything runs on MPS out of the box. 
For the heavier runs (DreamerV3 on DMControl, full MBPO training), expect:
- **CartPole/Pendulum**: 1-5 minutes on MPS
- **HalfCheetah/Hopper**: 10-30 minutes on MPS
- **DreamerV3 (1M steps)**: 2-4 hours on MPS (or ~30min with CUDA)

## What each algorithm does (in one sentence)

- **DreamerV3**: Learns a recurrent world model from pixels, then optimizes a policy by planning imagined trajectories in the latent space.
- **MuZero**: Learns a predictive model of an unknown environment and uses MCTS with the model for action selection — no dynamics function needed.
- **MBPO**: Trains an ensemble of dynamics models, generates virtual rollouts, and optimizes the policy on synthetic data while keeping a safety buffer from real data.
- **PETS**: Uses a Bayesian ensemble of neural network dynamics models to estimate uncertainty during rollouts, preventing the policy from exploiting model errors.
- **TD-MPC2**: Learns a latent dynamics model with TD loss, then at each step samples hundreds of action trajectories and picks the best one via cross-entropy method.
- **JEPA**: Self-supervised world model that predicts future latent states without reconstructing pixels. Trained on N-body physics to see if it learns conservation laws.

## Results

Training logs and plots are in `logs/` and `results/`. Run `python plot_results.py` to regenerate the comparison grid from any log files.

Key takeaways from training runs:
- **MuZero** solves CartPole reliably within 40k steps
- **TD-MPC2** converges fast on Pendulum but needs careful hyperparameter tuning
- **MBPO** handles MuJoCo environments well but needs ensemble diversity
- **DreamerV3** is the most complex but handles pixel inputs natively
- **JEPA** on N-body shows promising latent dynamics learning with energy conservation properties

## Citations / References

- **DreamerV3**: https://arxiv.org/abs/2301.04104
- **MuZero**: https://arxiv.org/abs/2008.01655
- **EfficientZero**: https://arxiv.org/abs/2102.06177
- **MBPO**: https://arxiv.org/abs/1909.11956
- **PETS**: https://arxiv.org/abs/1703.04730
- **TD-MPC2**: https://arxiv.org/abs/2310.16828
- **JEPA**: https://arxiv.org/abs/2205.09113

## License

MIT
