import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from collections import deque
import random
import time

class Config:
    env_id = "Pendulum-v1"
    latent_dim = 50
    hidden_dim = 256
    total_steps = 30_000
    seed_steps = 1000
    batch_size = 256
    train_freq = 2
    horizon = 5
    num_samples = 512
    num_elites = 64
    temperature = 0.5
    momentum = 0.1
    lr = 3e-4
    gamma = 0.99
    tau = 0.01
    buffer_size = 1_000_000
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42
    eval_episodes = 10
    eval_freq = 5000

class Encoder(nn.Module):
    def __init__(self, obs_dim, latent_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, obs):
        return self.net(obs)

class Dynamics(nn.Module):
    def __init__(self, latent_dim, action_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
    def forward(self, z, a):
        return self.net(torch.cat([z, a], dim=-1))

class Reward(nn.Module):
    def __init__(self, latent_dim, action_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, z, a):
        return self.net(torch.cat([z, a], dim=-1))

class QFunction(nn.Module):
    def __init__(self, latent_dim, action_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, z, a):
        return self.net(torch.cat([z, a], dim=-1))

class TDMPC:
    def __init__(self, obs_dim, action_dim, config):
        self.config = config
        self.action_dim = action_dim
        self.encoder = Encoder(obs_dim, config.latent_dim, config.hidden_dim).to(config.device)
        self.dynamics = Dynamics(config.latent_dim, action_dim, config.hidden_dim).to(config.device)
        self.reward = Reward(config.latent_dim, action_dim, config.hidden_dim).to(config.device)
        self.q = QFunction(config.latent_dim, action_dim, config.hidden_dim).to(config.device)
        self.encoder_target = Encoder(obs_dim, config.latent_dim, config.hidden_dim).to(config.device)
        self.encoder_target.load_state_dict(self.encoder.state_dict())
        self.q_target = QFunction(config.latent_dim, action_dim, config.hidden_dim).to(config.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) +
            list(self.dynamics.parameters()) +
            list(self.reward.parameters()) +
            list(self.q.parameters()),
            lr=config.lr
        )
        self.prev_mean = None

    def plan(self, obs, action_low, action_high):
        z = self.encoder(torch.FloatTensor(obs).unsqueeze(0).to(self.config.device))
        if self.prev_mean is None:
            mean = torch.zeros(self.config.horizon, self.action_dim).to(self.config.device)
        else:
            mean = torch.cat([self.prev_mean[1:], torch.zeros(1, self.action_dim).to(self.config.device)])
        std = 2 * torch.ones(self.config.horizon, self.action_dim).to(self.config.device)
        noise = torch.randn(self.config.num_samples, self.config.horizon, self.action_dim).to(self.config.device)
        actions = mean + std * noise
        actions = torch.clamp(actions, action_low, action_high)
        z_expanded = z.repeat(self.config.num_samples, 1)
        returns = torch.zeros(self.config.num_samples).to(self.config.device)
        z_t = z_expanded
        for t in range(self.config.horizon):
            a_t = actions[:, t]
            z_t = self.dynamics(z_t, a_t)
            r_t = self.reward(z_t, a_t)
            returns += r_t.squeeze() * (self.config.gamma ** t)
        returns += self.q(z_t, torch.zeros(self.config.num_samples, self.action_dim).to(self.config.device)).squeeze() * (self.config.gamma ** self.config.horizon)
        elite_idx = torch.topk(returns, self.config.num_elites).indices
        elite_actions = actions[elite_idx]
        new_mean = elite_actions.mean(dim=0)
        self.prev_mean = self.config.momentum * mean + (1 - self.config.momentum) * new_mean
        return self.prev_mean[0].cpu().numpy()

    def update(self, batch):
        obs, actions, rewards, next_obs, dones = batch
        obs = torch.FloatTensor(obs).to(self.config.device)
        actions = torch.FloatTensor(actions).to(self.config.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.config.device)
        next_obs = torch.FloatTensor(next_obs).to(self.config.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.config.device)
        z = self.encoder(obs)
        z_next = self.dynamics(z, actions)
        pred_reward = self.reward(z, actions)
        with torch.no_grad():
            z_target = self.encoder_target(next_obs)
            q_target = self.q_target(z_target, torch.zeros_like(actions))
            target = rewards + self.config.gamma * (1 - dones) * q_target
        q_pred = self.q(z, actions)
        consistency_loss = F.mse_loss(z_next, z_target.detach())
        reward_loss = F.mse_loss(pred_reward, rewards)
        value_loss = F.mse_loss(q_pred, target)
        loss = consistency_loss + reward_loss + value_loss
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 10.0)
        torch.nn.utils.clip_grad_norm_(self.dynamics.parameters(), 10.0)
        torch.nn.utils.clip_grad_norm_(self.reward.parameters(), 10.0)
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.optimizer.step()
        for param, target_param in zip(self.encoder.parameters(), self.encoder_target.parameters()):
            target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data)
        for param, target_param in zip(self.q.parameters(), self.q_target.parameters()):
            target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data)
        return loss.item()

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return (
            np.array(obs),
            np.array(actions),
            np.array(rewards),
            np.array(next_obs),
            np.array(dones)
        )
    def __len__(self):
        return len(self.buffer)
def evaluate(env, agent, config):
    total_rewards = []
    for _ in range(config.eval_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        agent.prev_mean = None
        while not done:
            action = agent.plan(obs, env.action_space.low[0], env.action_space.high[0])
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        total_rewards.append(episode_reward)
    return np.mean(total_rewards)

def main():
    config = Config()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    env = gym.make(config.env_id)
    env.reset(seed=config.seed)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = TDMPC(obs_dim, action_dim, config)
    buffer = ReplayBuffer(config.buffer_size)
    obs, _ = env.reset()
    episode_reward = 0
    best_eval = -float('inf')
    start_time = time.time()
    for step in range(config.total_steps):
        if step < config.seed_steps:
            action = env.action_space.sample()
        else:
            action = agent.plan(obs, env.action_space.low[0], env.action_space.high[0])
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.push(obs, action, reward, next_obs, done)
        episode_reward += reward
        if done:
            episode_reward = 0
            obs, _ = env.reset()
            agent.prev_mean = None
        else:
            obs = next_obs
        if step > config.seed_steps and step % config.train_freq == 0:
            batch = buffer.sample(config.batch_size)
            loss = agent.update(batch)
        if step > 0 and step % config.eval_freq == 0:
            eval_reward = evaluate(env, agent, config)
            elapsed = time.time() - start_time
            if eval_reward > best_eval:
                best_eval = eval_reward
            print(f"step {step}/{config.total_steps} | Eval: {eval_reward:.1f} | Best: {best_eval:.1f} | Time: {elapsed/60:.1f}m")
    env.close()
    print(f"\nbest: {best_eval:.1f}")
if __name__ == "__main__":
    main()