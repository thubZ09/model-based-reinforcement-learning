import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from collections import deque
import random
from typing import List, Tuple
import time

class Config:
    env_id = "CartPole-v1"
    max_episode_steps = 500
    ensemble_size = 5
    hidden_size = 128
    learning_rate = 0.001
    buffer_size = 10_000
    initial_random_steps = 1000
    planning_horizon = 15
    cem_iterations = 5
    cem_population = 500
    cem_elite_frac = 0.1
    total_steps = 20_000
    model_train_freq = 250
    model_train_epochs = 5
    batch_size = 256
    eval_freq = 1000
    device = "cuda" if torch.cuda.is_available() else "cpu"
    eval_episodes = 5
    print_freq = 500
    seed = 42

class EnsembleDynamics(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int, ensemble_size: int):
        super().__init__()
        self.ensemble_size = ensemble_size

        self.networks = nn.ModuleList([
            self._build_network(state_dim, action_dim, hidden_size)
            for _ in range(ensemble_size)
        ])

    def _build_network(self, state_dim: int, action_dim: int, hidden_size: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, state_dim + 1)
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if action.dim() == 1:
            action = F.one_hot(action.long(), num_classes=2).float()

        x = torch.cat([state, action], dim=-1)
        predictions = []
        for network in self.networks:
            pred = network(x)
            predictions.append(pred)
        predictions = torch.stack(predictions)
        next_states = predictions[..., :-1]
        rewards = predictions[..., -1:]

        return next_states, rewards
    
    def sample(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        next_states, rewards = self.forward(state, action)
        batch_size = state.shape[0]
        indices = torch.randint(0, self.ensemble_size, (batch_size,), device=state.device)
        selected_next_states = next_states[indices, torch.arange(batch_size)]
        selected_rewards = rewards[indices, torch.arange(batch_size)]

        return selected_next_states, selected_rewards

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
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

def cem_planning(
    model: EnsembleDynamics,
    state: np.ndarray,
    horizon: int,
    iterations: int,
    population: int,
    elite_frac: float,
    action_dim: int,
    device: str
) -> int:

    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    elite_size = int(population * elite_frac)
    action_probs = torch.ones(horizon, action_dim, device=device) / action_dim

    for _ in range(iterations):
        action_sequences = torch.multinomial(
            action_probs.repeat(population, 1, 1).view(-1, action_dim),
            num_samples=1
        ).view(population, horizon).to(device)

        returns = torch.zeros(population, device=device)
        current_states = state_tensor.repeat(population, 1)

        for t in range(horizon):
            actions = action_sequences[:, t]
            next_states, rewards = model.sample(current_states, actions)
            returns += rewards.squeeze()
            current_states = next_states

        elite_indices = returns.topk(elite_size).indices
        elite_actions = action_sequences[elite_indices]

        for t in range(horizon):
            elite_actions_t = elite_actions[:, t]
            counts = torch.bincount(elite_actions_t, minlength=action_dim).float()
            action_probs[t] = counts / counts.sum()
    
    best_idx = returns.argmax()
    best_action = action_sequences[best_idx, 0].item()

    return best_action

def train_dynamics_model(
    model: EnsembleDynamics,
    buffer: ReplayBuffer,
    optimizer: torch.optim.Optimizer,
    config: Config,
    epochs: int = 5
) -> float:

    model.train()
    total_loss = 0.0
    num_batches = 0

    for epoch in range(epochs):
        for _ in range(len(buffer) // config.batch_size):
            states, actions, rewards, next_states, _ = buffer.sample(config.batch_size)

            states = torch.FloatTensor(states).to(config.device)
            actions = torch.LongTensor(actions).to(config.device)
            rewards = torch.FloatTensor(rewards).to(config.device)
            next_states = torch.FloatTensor(next_states).to(config.device)
            pred_next_states, pred_rewards = model.forward(states, actions)

            state_loss = F.mse_loss(
                pred_next_states,
                next_states.unsqueeze(0).expand(config.ensemble_size, -1, -1)
            )
            reward_loss = F.mse_loss(
                pred_rewards,   
                rewards.unsqueeze(0).unsqueeze(-1).expand(config.ensemble_size, -1, -1)
            )
            loss = state_loss + reward_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        
    return total_loss / num_batches

def evaluate_policy(model: EnsembleDynamics, config: Config) -> float:
    model.eval()
    env = gym.make(config.env_id)
    total_rewards = []

    for _ in range(config.eval_episodes):
        state, _ = env.reset()
        episode_reward = 0

        for _ in range(config.max_episode_steps):
            with torch.no_grad():
                action = cem_planning(
                    model=model,
                    state=state,
                    horizon=config.planning_horizon,
                    iterations=config.cem_iterations,
                    population=config.cem_population,
                    elite_frac=config.cem_elite_frac,
                    action_dim=2,
                    device=config.device
                )
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            if terminated or truncated:
                break
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
    action_dim = env.action_space.n

    model = EnsembleDynamics(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_size=config.hidden_size,
        ensemble_size=config.ensemble_size
    ).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    buffer = ReplayBuffer(capacity=config.buffer_size)


       