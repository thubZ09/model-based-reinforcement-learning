"""
Paper: "When to Trust Your Model - Model-Based Policy Optimization"
Authors: Janner et al. (2019)
Link: https://arxiv.org/abs/1906.08253

Hybrid approach combining model-based and model-free RL
> Learn dynamics model from real environment
> Generate synthetic rollouts using model
> Train SAC policy on mix of real + synthetic data
> More sample efficient than pure model-free
> More robust than pure model-based
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from collections import deque
import random
from typing import Tuple
import time

class Config:
    env_id = "HalfCheetah-v4"
    ensemble_size = 7
    model_hidden_size = 200
    model_lr = 0.001
    elite_size = 5
    actor_hidden_size = 256
    critic_hidden_size = 256
    sac_lr = 0.0003
    alpha = 0.2
    gamma = 0.99
    tau = 0.005
    rollout_length = 1
    rollout_batch_size = 400
    real_ratio = 0.05
    model_retain_epochs = 5
    total_steps = 300_000
    model_train_freq = 250
    policy_train_freq = 1
    eval_freq = 5000
    env_buffer_size = 1_000_000
    model_buffer_size = 1_000_000
    batch_size = 256
    warmup_steps = 5000
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42

class EnsembleDynamics(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int, ensemble_size: int):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.networks = nn.ModuleList([
            self._build_network(state_dim, action_dim, hidden_size)
            for _ in range(ensemble_size)
        ])
        self.max_logvar = nn.Parameter(torch.ones(1, state_dim) * 0.5)
        self.min_logvar = nn.Parameter(torch.ones(1, state_dim) * -10)

    def _build_network(self, state_dim: int, action_dim: int, hidden_size: int):
        return nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2 * state_dim + 1)
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        x = torch.cat([state, action], dim=-1)
        outputs = []
        for network in self.networks:
            out = network(x)
            mean = out[..., :-1].chunk(2, dim=-1)[0]
            logvar = out[..., :-1].chunk(2, dim=-1)[1]
            logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
            logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
            outputs.append(torch.cat([mean, logvar, out[..., -1:]], dim=-1))
        return torch.stack(outputs)

    def loss(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, reward: torch.Tensor):
        outputs = self.forward(state, action)
        delta_target = next_state - state

        means = outputs[..., :state.shape[-1]]
        logvars = outputs[..., state.shape[-1]:2*state.shape[-1]]
        reward_preds = outputs[..., -1]

        inv_vars = torch.exp(-logvars)
        mse_losses = ((means - delta_target.unsqueeze(0)) ** 2) * inv_vars
        var_losses = logvars

        state_loss = (mse_losses + var_losses).mean(dim=(1, 2))
        reward_loss = F.mse_loss(reward_preds, reward.unsqueeze(0).expand(self.ensemble_size, -1), reduction='none').mean(dim=1)

        total_loss = (state_loss + reward_loss).sum()
        return total_loss, state_loss.mean().item()

    def predict(self, state: torch.Tensor, action: torch.Tensor, deterministic: bool = False):
        with torch.no_grad():
            outputs = self.forward(state, action)
            idx = torch.randint(0, self.ensemble_size, (state.shape[0],), device=state.device)

            means = outputs[..., :state.shape[-1]]
            logvars = outputs[..., state.shape[-1]:2*state.shape[-1]]
            reward_preds = outputs[..., -1]

            selected_means = means[idx, torch.arange(state.shape[0])]
            selected_logvars = logvars[idx, torch.arange(state.shape[0])]
            selected_rewards = reward_preds[idx, torch.arange(state.shape[0])]

            if deterministic:
                next_state = state + selected_means
            else:
                std = torch.exp(0.5 * selected_logvars)
                next_state = state + selected_means + std * torch.randn_like(selected_means)

            return next_state, selected_rewards


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int, action_scale: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.mean = nn.Linear(hidden_size, action_dim)
        self.log_std = nn.Linear(hidden_size, action_dim)
        self.action_scale = action_scale

    def forward(self, state: torch.Tensor):
        x = self.net(state)
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(-20, 2)
        return mean, log_std

    def sample(self, state: torch.Tensor):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x = normal.rsample()
        action = torch.tanh(x)
        log_prob = normal.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action * self.action_scale, log_prob


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        x = torch.cat([state, action], dim=-1)
        return self.q1(x), self.q2(x)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )

    def __len__(self):
        return len(self.buffer)


def train_dynamics_model(model, buffer, optimizer, config, epochs=5):
    model.train()
    total_loss = 0
    num_batches = 0

    for epoch in range(epochs):
        for _ in range(len(buffer) // config.batch_size):
            states, actions, rewards, next_states, _ = buffer.sample(config.batch_size)

            states = torch.FloatTensor(states).to(config.device)
            actions = torch.FloatTensor(actions).to(config.device)
            rewards = torch.FloatTensor(rewards).to(config.device)
            next_states = torch.FloatTensor(next_states).to(config.device)

            loss, state_loss = model.loss(states, actions, next_states, rewards)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += state_loss
            num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0


def update_sac(actor, critic, critic_target, actor_optimizer, critic_optimizer,
               env_buffer, model_buffer, config):

    env_batch_size = int(config.batch_size * config.real_ratio)
    model_batch_size = config.batch_size - env_batch_size

    env_states, env_actions, env_rewards, env_next_states, env_dones = env_buffer.sample(env_batch_size)
    model_states, model_actions, model_rewards, model_next_states, model_dones = model_buffer.sample(model_batch_size)

    states = np.concatenate([env_states, model_states])
    actions = np.concatenate([env_actions, model_actions])
    rewards = np.concatenate([env_rewards, model_rewards])
    next_states = np.concatenate([env_next_states, model_next_states])
    dones = np.concatenate([env_dones, model_dones])

    states = torch.FloatTensor(states).to(config.device)
    actions = torch.FloatTensor(actions).to(config.device)
    rewards = torch.FloatTensor(rewards).unsqueeze(1).to(config.device)
    next_states = torch.FloatTensor(next_states).to(config.device)
    dones = torch.FloatTensor(dones).unsqueeze(1).to(config.device)

    with torch.no_grad():
        next_actions, next_log_probs = actor.sample(next_states)
        q1_next, q2_next = critic_target(next_states, next_actions)
        q_next = torch.min(q1_next, q2_next) - config.alpha * next_log_probs
        q_target = rewards + config.gamma * (1 - dones) * q_next

    q1, q2 = critic(states, actions)
    critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    new_actions, log_probs = actor.sample(states)
    q1_new, q2_new = critic(states, new_actions)
    q_new = torch.min(q1_new, q2_new)
    actor_loss = (config.alpha * log_probs - q_new).mean()

    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    for param, target_param in zip(critic.parameters(), critic_target.parameters()):
        target_param.data.copy_(config.tau * param.data + (1 - config.tau) * target_param.data)

    return actor_loss.item(), critic_loss.item()


def generate_model_rollouts(model, actor, env_buffer, model_buffer, config):
    states, _, _, _, _ = env_buffer.sample(config.rollout_batch_size)
    states = torch.FloatTensor(states).to(config.device)

    for _ in range(config.rollout_length):
        with torch.no_grad():
            actions, _ = actor.sample(states)
            next_states, rewards = model.predict(states, actions)

        for i in range(states.shape[0]):
            model_buffer.push(
                states[i].cpu().numpy(),
                actions[i].cpu().numpy(),
                rewards[i].item(),
                next_states[i].cpu().numpy(),
                0
            )
        states = next_states

def evaluate_policy(actor, config, num_episodes=10):
    env = gym.make(config.env_id)
    total_rewards = []

    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(config.device)
            with torch.no_grad():
                action, _ = actor.sample(state_tensor)
            state, reward, terminated, truncated, _ = env.step(action.cpu().numpy()[0])
            episode_reward += reward
            done = terminated or truncated

        total_rewards.append(episode_reward)

    env.close()
    return np.mean(total_rewards)


def main():
    config = Config()

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    env = gym.make(config.env_id)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_scale = float(env.action_space.high[0])

    model = EnsembleDynamics(state_dim, action_dim, config.model_hidden_size, config.ensemble_size).to(config.device)
    actor = Actor(state_dim, action_dim, config.actor_hidden_size, action_scale).to(config.device)
    critic = Critic(state_dim, action_dim, config.critic_hidden_size).to(config.device)
    critic_target = Critic(state_dim, action_dim, config.critic_hidden_size).to(config.device)
    critic_target.load_state_dict(critic.state_dict())

    model_optimizer = torch.optim.Adam(model.parameters(), lr=config.model_lr)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.sac_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=config.sac_lr)

    env_buffer = ReplayBuffer(config.env_buffer_size)
    model_buffer = ReplayBuffer(config.model_buffer_size)

    state, _ = env.reset()
    episode_reward = 0
    best_eval = -float('inf')
    start_time = time.time()

    for step in range(config.total_steps):
        if step < config.warmup_steps:
            action = env.action_space.sample()
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(config.device)
            with torch.no_grad():
                action, _ = actor.sample(state_tensor)
            action = action.cpu().numpy()[0]

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        env_buffer.push(state, action, reward, next_state, done)
        episode_reward += reward

        if done:
            episode_reward = 0
            state, _ = env.reset()
        else:
            state = next_state

        if step > config.warmup_steps and step % config.model_train_freq == 0:
            model_loss = train_dynamics_model(model, env_buffer, model_optimizer, config, epochs=config.model_retain_epochs)
            generate_model_rollouts(model, actor, env_buffer, model_buffer, config)

        if step > config.warmup_steps and len(model_buffer) > config.batch_size:
            actor_loss, critic_loss = update_sac(actor, critic, critic_target, actor_optimizer,
                                                  critic_optimizer, env_buffer, model_buffer, config)

        if step > 0 and step % config.eval_freq == 0:
            eval_reward = evaluate_policy(actor, config)
            elapsed = time.time() - start_time

            if eval_reward > best_eval:
                best_eval = eval_reward

            print(f"Step {step}/{config.total_steps} | Eval: {eval_reward:.1f} | Best: {best_eval:.1f} | Time: {elapsed/60:.1f}m")

    env.close()
    print(f"\ntraining complete! best: {best_eval:.1f}")

if __name__ == "__main__":
    main()