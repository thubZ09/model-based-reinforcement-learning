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
    env_id = "CartPole-v1"
    max_episode_steps = 500    
    ensemble_size = 5
    hidden_size = 256  
    learning_rate = 0.001    
    buffer_size = 50_000  
    initial_random_steps = 2500     
    planning_horizon = 12
    num_candidates = 500      
    epsilon_start = 0.3  
    epsilon_end = 0.05   
    epsilon_decay_steps = 15_000    
    total_steps = 30_000  
    model_train_freq = 250
    model_train_epochs = 10 
    batch_size = 256    
    eval_freq = 2000
    eval_episodes = 10     
    device = "cuda" if torch.cuda.is_available() else "cpu"    
    print_freq = 1000    
    seed = 42

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
            nn.Linear(hidden_size, state_dim + 1)
        )
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if action.dim() == 1:
            action = F.one_hot(action.long(), num_classes=self.action_dim).float()
        
        x = torch.cat([state, action], dim=-1)
        
        predictions = []
        for network in self.networks:
            pred = network(x)
            predictions.append(pred)
        
        predictions = torch.stack(predictions)
        delta_states = predictions[..., :-1]
        rewards = predictions[..., -1:]
        
        next_states = state.unsqueeze(0) + delta_states
        
        return next_states, rewards
    
    def predict_with_uncertainty(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        next_states, rewards = self.forward(state, action)
        
        mean_next_state = next_states.mean(dim=0)
        mean_reward = rewards.mean(dim=0)        
        uncertainty = next_states.std(dim=0).mean(dim=-1, keepdim=True)
        
        return mean_next_state, mean_reward, uncertainty

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

def random_shooting_planner(
    model: EnsembleDynamics,
    state: np.ndarray,
    horizon: int,
    num_candidates: int,
    action_dim: int,
    device: str,
    uncertainty_penalty: float = 0.5
) -> int:
    
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    action_sequences = torch.randint(
        0, action_dim, 
        (num_candidates, horizon), 
        device=device
    )    
    returns = torch.zeros(num_candidates, device=device)
    uncertainty_sum = torch.zeros(num_candidates, device=device)
    
    current_states = state_tensor.repeat(num_candidates, 1)
    gamma = 0.99
    
    for t in range(horizon):
        actions = action_sequences[:, t]
        
        with torch.no_grad():
            next_states, rewards, uncertainty = model.predict_with_uncertainty(
                current_states, actions
            )        
        returns += (gamma ** t) * rewards.squeeze()
        uncertainty_sum += uncertainty.squeeze()
        
        current_states = next_states

    scores = returns - uncertainty_penalty * uncertainty_sum    
    best_idx = scores.argmax()
    best_action = action_sequences[best_idx, 0].item()
    
    return best_action

def train_dynamics_model(
    model: EnsembleDynamics,
    buffer: ReplayBuffer,
    optimizer: torch.optim.Optimizer,
    config: Config,
    epochs: int = 10
) -> float:
    
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for epoch in range(epochs):
        for _ in range(min(50, len(buffer) // config.batch_size)): 
            states, actions, rewards, next_states, _ = buffer.sample(config.batch_size)
            
            states = torch.FloatTensor(states).to(config.device)
            actions = torch.LongTensor(actions).to(config.device)
            rewards = torch.FloatTensor(rewards).to(config.device)
            next_states = torch.FloatTensor(next_states).to(config.device)
            
            pred_next_states, pred_rewards = model.forward(states, actions)
            
            target_next_states = next_states.unsqueeze(0).expand(config.ensemble_size, -1, -1)
            target_rewards = rewards.unsqueeze(0).unsqueeze(-1).expand(config.ensemble_size, -1, -1)
            
            state_loss = F.mse_loss(pred_next_states, target_next_states)
            reward_loss = F.mse_loss(pred_rewards, target_rewards)
            
            l2_reg = 0.0
            for param in model.parameters():
                l2_reg += torch.norm(param)
            
            loss = state_loss + 3.0 * reward_loss + 0.0001 * l2_reg
            
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
    
    for _ in range(config.eval_episodes):
        state, _ = env.reset()
        episode_reward = 0
        
        for _ in range(config.max_episode_steps):
            with torch.no_grad():
                action = random_shooting_planner(
                    model=model,
                    state=state,
                    horizon=config.planning_horizon,
                    num_candidates=config.num_candidates,
                    action_dim=2,
                    device=config.device,
                    uncertainty_penalty=0.5
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
    env.reset(seed=config.seed)
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
    episode_rewards = []
    current_episode_reward = 0
    best_eval_reward = 0
    state, _ = env.reset()
    start_time = time.time()
    
    for step in range(config.total_steps):
        epsilon = config.epsilon_start - (config.epsilon_start - config.epsilon_end) * min(step / config.epsilon_decay_steps, 1.0)
        
        if step < config.initial_random_steps or random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = random_shooting_planner(
                    model=model,
                    state=state,
                    horizon=config.planning_horizon,
                    num_candidates=config.num_candidates,
                    action_dim=action_dim,
                    device=config.device,
                    uncertainty_penalty=0.5
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
                    
        if step > 0 and step % config.eval_freq == 0:
            eval_reward = evaluate_policy(model, config)
            
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
            
            elapsed = time.time() - start_time

            if len(episode_rewards) > 0:
                recent = episode_rewards[-20:]
            
            if best_eval_reward >= 195:
                break    
    env.close()

if __name__ == "__main__":
    main()          



       