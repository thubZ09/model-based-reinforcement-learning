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
    env_id = "Pendulum-v1"
    max_episode_steps = 200    
    ensemble_size = 5           
    hidden_size = 200           
    learning_rate = 0.001    
    buffer_size = 100_000
    initial_random_steps = 1000      
    planning_horizon = 25      
    cem_iterations = 5         
    cem_population = 400       
    cem_elite_frac = 0.1        
    cem_alpha = 0.25               
    total_steps = 10_000
    seed = 42       
    model_train_freq = 250     
    model_train_epochs = 5
    batch_size = 256    
    eval_freq = 1000
    eval_episodes = 10
    normalize_rewards = True    
    device = "cuda" if torch.cuda.is_available() else "cpu"    
    print_freq = 500
                    
class EnsembleDynamics(nn.Module):    
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int, ensemble_size: int):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.state_dim = state_dim
        self.action_dim = action_dim        
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
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, state_dim + 1) 
        )
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([state, action], dim=-1)
        predictions = []
        for network in self.networks:
            pred = network(x)
            predictions.append(pred)
        predictions = torch.stack(predictions)         
        delta_states = predictions[..., :-1]
        rewards = predictions[..., -1:        
        next_states = state.unsqueeze(0) + delta_states
        return next_states, rewards
    def sample(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        next_states, rewards = self.forward(state, action)        
        batch_size = state.shape[0]
        indices = torch.randint(0, self.ensemble_size, (batch_size,), device=state.device)
        selected_next_states = next_states[indices, torch.arange(batch_size)]
        selected_rewards = rewards[indices, torch.arange(batch_size)]
        return selected_next_states, selected_rewards

class ReplayBuffer:
    def __init__(self, capacity: int, normalize_rewards: bool = True):
        self.buffer = deque(maxlen=capacity)
        self.normalize_rewards = normalize_rewards        
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_history = deque(maxlen=10000)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        if self.normalize_rewards:
            self.reward_history.append(reward)
            if len(self.reward_history) > 100:
                self.reward_mean = np.mean(self.reward_history)
                self.reward_std = np.std(self.reward_history) + 1e-6
    def sample(self, batch_size: int) -> Tuple:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        if self.normalize_rewards:
            rewards = [(r - self.reward_mean) / self.reward_std for r in rewards]
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    def __len__(self):
        return len(self.buffer)

def cem_planning_continuous(
    model: EnsembleDynamics,
    state: np.ndarray,
    horizon: int,
    iterations: int,
    population: int,
    elite_frac: float,
    action_dim: int,
    action_low: float,
    action_high: float,
    alpha: float,
    device: str
) -> np.ndarray:
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    elite_size = int(population * elite_frac)
    mean = torch.zeros(horizon, action_dim, device=device)
    std = torch.ones(horizon, action_dim, device=device) * (action_high - action_low) / 2
    for iteration in range(iterations):
        noise = torch.randn(population, horizon, action_dim, device=device)
        action_sequences = mean + std * noise
        action_sequences = torch.clamp(action_sequences, action_low, action_high)        
        returns = torch.zeros(population, device=device)
        current_states = state_tensor.repeat(population, 1)
        gamma = 0.99
        for t in range(horizon):
            actions = action_sequences[:, t, :]
            with torch.no_grad():
                next_states, rewards = model.sample(current_states, actions)
            returns += (gamma ** t) * rewards.squeeze()
            current_states = next_states
        elite_indices = returns.topk(elite_size).indices
        elite_actions = action_sequences[elite_indices]        
        new_mean = elite_actions.mean(dim=0)
        new_std = elite_actions.std(dim=0) + 1e-6         
        mean = alpha * mean + (1 - alpha) * new_mean
        std = alpha * std + (1 - alpha) * new_std
    best_action = mean[0].cpu().numpy()
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
        for _ in range(min(50, len(buffer) // config.batch_size)):
            states, actions, rewards, next_states, _ = buffer.sample(config.batch_size)
            states = torch.FloatTensor(states).to(config.device)
            actions = torch.FloatTensor(actions).to(config.device)
            rewards = torch.FloatTensor(rewards).unsqueeze(-1).to(config.device)
            next_states = torch.FloatTensor(next_states).to(config.device)            
            pred_next_states, pred_rewards = model.forward(states, actions)            
            target_next_states = next_states.unsqueeze(0).expand(config.ensemble_size, -1, -1)
            target_rewards = rewards.unsqueeze(0).expand(config.ensemble_size, -1, -1)
            state_loss = F.mse_loss(pred_next_states, target_next_states)
            reward_loss = F.mse_loss(pred_rewards, target_rewards)            
            loss = state_loss + 2.0 * reward_loss            
            optimizer.zero_grad()
            loss.backward()            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
    return total_loss / num_batches if num_batches > 0 else 0.0

def evaluate_policy(model: EnsembleDynamics, config: Config) -> float:    
    model.eval()
    env = gym.make(config.env_id)
    total_rewards = []
    action_low = env.action_space.low[0]
    action_high = env.action_space.high[0]
    for _ in range(config.eval_episodes):
        state, _ = env.reset()
        episode_reward = 0
        for _ in range(config.max_episode_steps):
            with torch.no_grad():
                action = cem_planning_continuous(
                    model=model,
                    state=state,
                    horizon=config.planning_horizon,
                    iterations=config.cem_iterations,
                    population=config.cem_population,
                    elite_frac=config.cem_elite_frac,
                    action_dim=1,  
                    action_low=action_low,
                    action_high=action_high,
                    alpha=config.cem_alpha,
                    device=config.device
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
    env.reset(seed=config.seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_low = env.action_space.low[0]
    action_high = env.action_space.high[0]
    model = EnsembleDynamics(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_size=config.hidden_size,
        ensemble_size=config.ensemble_size
    ).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    buffer = ReplayBuffer(
        capacity=config.buffer_size,
        normalize_rewards=config.normalize_rewards
    )
    episode_rewards = []
    current_episode_reward = 0
    best_eval_reward = -float('inf')
    state, _ = env.reset()
    start_time = time.time()
    for step in range(config.total_steps):
        if step < config.initial_random_steps:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = cem_planning_continuous(
                    model=model,
                    state=state,
                    horizon=config.planning_horizon,
                    iterations=config.cem_iterations,
                    population=config.cem_population,
                    elite_frac=config.cem_elite_frac,
                    action_dim=action_dim,
                    action_low=action_low,
                    action_high=action_high,
                    alpha=config.cem_alpha,
                    device=config.device
                )
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.push(state, action, reward, next_state, done)
        current_episode_reward += reward
        if done:
            episode_rewards.append(current_episode_reward)
            current_episode_reward = 0
            state, _ = env.reset()
        else:
            state = next_state
        if step > config.initial_random_steps and step % config.model_train_freq == 0:
            loss = train_dynamics_model(
                model=model,
                buffer=buffer,
                optimizer=optimizer,
                config=config,
                epochs=config.model_train_epochs
            )
            if step % config.print_freq == 0:
                print(f"step {step}: model loss = {loss:.4f}")
        if step > 0 and step % config.eval_freq == 0:
            eval_reward = evaluate_policy(model, config)
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                print(f"new best: {eval_reward:.2f}")
            elapsed = time.time() - start_time
            episodes_completed = len(episode_rewards)
            if best_eval_reward >= -200:
                break
    env.close()
if __name__ == "__main__":
    main()