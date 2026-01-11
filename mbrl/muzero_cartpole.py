"""
Paper - "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model"
Authors: Schrittwieser et al. (2020)
Link: https://arxiv.org/abs/1911.08265

Simplified version for discrete action spaces
> Learned dynamics model with value equivalence
> MCTS planning in latent space
> No explicit observation reconstruction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from collections import deque
import random
import time
import math

class Config:
    env_id = "CartPole-v1"
    latent_dim = 64
    hidden_dim = 128    
    num_simulations = 50
    c1 = 1.25
    c2 = 19652
    discount = 0.99    
    total_steps = 20_000
    batch_size = 128
    unroll_steps = 5
    td_steps = 5    
    lr = 0.001
    weight_decay = 1e-4    
    buffer_size = 10_000    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42
    eval_episodes = 10
    eval_freq = 2000

class RepresentationNetwork(nn.Module):
    def __init__(self, obs_dim, latent_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.Tanh()
        )
    
    def forward(self, obs):
        return self.net(obs)

class DynamicsNetwork(nn.Module):
    def __init__(self, latent_dim, action_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.Tanh()
        )
        self.reward_head = nn.Linear(latent_dim, 1)
    
    def forward(self, state, action):
        action_one_hot = F.one_hot(action.long(), num_classes=2).float()
        x = torch.cat([state, action_one_hot], dim=-1)
        next_state = self.net(x)
        reward = self.reward_head(next_state)
        return next_state, reward

class PredictionNetwork(nn.Module):
    def __init__(self, latent_dim, action_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU()
        )
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, state):
        x = self.net(state)
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        return policy_logits, value

class Node:
    def __init__(self, prior):
        self.visit_count = 0
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.hidden_state = None
        self.reward = 0
    
    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

class MCTS:
    def __init__(self, config, representation, dynamics, prediction):
        self.config = config
        self.representation = representation
        self.dynamics = dynamics
        self.prediction = prediction
    
    def run(self, obs, num_actions):
        root = Node(0)
        
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.config.device)
        with torch.no_grad():
            root.hidden_state = self.representation(obs_tensor)
            policy_logits, value = self.prediction(root.hidden_state)
            policy = torch.softmax(policy_logits, dim=-1).cpu().numpy()[0]
        
        for action in range(num_actions):
            root.children[action] = Node(policy[action])
        
        for _ in range(self.config.num_simulations):
            node = root
            search_path = [node]
            
            while node.children:
                action, node = self.select_child(node)
                search_path.append(node)
            
            parent = search_path[-2]
            
            if node.hidden_state is None:
                with torch.no_grad():
                    node.hidden_state, reward = self.dynamics(
                        parent.hidden_state,
                        torch.tensor([action]).to(self.config.device)
                    )
                    node.reward = reward.cpu().item()
                    
                    policy_logits, value = self.prediction(node.hidden_state)
                    policy = torch.softmax(policy_logits, dim=-1).cpu().numpy()[0]
                    
                    for a in range(num_actions):
                        node.children[a] = Node(policy[a])
            
            self.backpropagate(search_path, value.cpu().item())
        
        visit_counts = np.array([root.children[a].visit_count for a in range(num_actions)])
        actions = list(range(num_actions))
        action = actions[np.argmax(visit_counts)]
        
        return action, root
    
    def select_child(self, node):
        max_ucb = -float('inf')
        max_action = None
        max_child = None
        
        for action, child in node.children.items():
            ucb = self.ucb_score(node, child)
            if ucb > max_ucb:
                max_ucb = ucb
                max_action = action
                max_child = child
        
        return max_action, max_child
    
    def ucb_score(self, parent, child):
        pb_c = math.log((parent.visit_count + self.config.c2 + 1) / self.config.c2) + self.config.c1
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)
        
        prior_score = pb_c * child.prior
        value_score = child.value()
        
        return prior_score + value_score
    
    def backpropagate(self, search_path, value):
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            value = node.reward + self.config.discount * value

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, obs, action, reward, policy, value):
        self.buffer.append((obs, action, reward, policy, value))
    
    def sample(self, batch_size, unroll_steps, td_steps):
        samples = []
        for _ in range(batch_size):
            idx = random.randint(0, len(self.buffer) - unroll_steps - td_steps)
            samples.append([self.buffer[idx + i] for i in range(unroll_steps + td_steps)])
        return samples
    
    def __len__(self):
        return len(self.buffer)

def train(representation, dynamics, prediction, optimizer, batch, config):
    total_loss = 0
    
    for trajectory in batch:
        obs = torch.FloatTensor([trajectory[0][0]]).to(config.device)
        
        hidden_state = representation(obs)
        
        for step in range(config.unroll_steps):
            action = torch.tensor([trajectory[step][1]]).to(config.device)
            target_value = sum([trajectory[step + i][2] * (config.discount ** i) 
                              for i in range(min(config.td_steps, len(trajectory) - step))])
            target_policy = trajectory[step][3]
            
            policy_logits, value = prediction(hidden_state)
            
            value_loss = F.mse_loss(value, torch.tensor([[target_value]]).to(config.device))
            policy_loss = F.cross_entropy(policy_logits, torch.tensor([target_policy]).to(config.device))
            
            if step < config.unroll_steps - 1:
                hidden_state, reward = dynamics(hidden_state, action)
                reward_loss = F.mse_loss(reward, torch.tensor([[trajectory[step][2]]]).to(config.device))
            else:
                reward_loss = 0
            
            total_loss += value_loss + policy_loss + reward_loss
    
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(
        list(representation.parameters()) + 
        list(dynamics.parameters()) + 
        list(prediction.parameters()), 
        10.0
    )
    optimizer.step()
    
    return total_loss.item() / len(batch)

def evaluate(env, mcts, config):
    total_rewards = []
    
    for _ in range(config.eval_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _ = mcts.run(obs, env.action_space.n)
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
    action_dim = env.action_space.n
    representation = RepresentationNetwork(obs_dim, config.latent_dim, config.hidden_dim).to(config.device)
    dynamics = DynamicsNetwork(config.latent_dim, action_dim, config.hidden_dim).to(config.device)
    prediction = PredictionNetwork(config.latent_dim, action_dim, config.hidden_dim).to(config.device)
    
    optimizer = torch.optim.Adam(
        list(representation.parameters()) + 
        list(dynamics.parameters()) + 
        list(prediction.parameters()),
        lr=config.lr,
        weight_decay=config.weight_decay
    )
    
    mcts = MCTS(config, representation, dynamics, prediction)
    buffer = ReplayBuffer(config.buffer_size)
    
    obs, _ = env.reset()
    episode_reward = 0
    best_eval = 0
    start_time = time.time()
    
    for step in range(config.total_steps):
        action, root = mcts.run(obs, action_dim)
        
        policy = np.array([root.children[a].visit_count for a in range(action_dim)])
        policy = policy / policy.sum()
        policy_action = np.argmax(policy)
        
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        buffer.push(obs, action, reward, policy_action, root.value())
        episode_reward += reward
        
        if done:
            episode_reward = 0
            obs, _ = env.reset()
        else:
            obs = next_obs
        
        if len(buffer) > config.batch_size + config.unroll_steps + config.td_steps:
            batch = buffer.sample(config.batch_size, config.unroll_steps, config.td_steps)
            loss = train(representation, dynamics, prediction, optimizer, batch, config)
        
        if step > 0 and step % config.eval_freq == 0:
            eval_reward = evaluate(env, mcts, config)
            elapsed = time.time() - start_time
            
            if eval_reward > best_eval:
                best_eval = eval_reward
            
            print(f"Step {step}/{config.total_steps} | Eval: {eval_reward:.1f} | Best: {best_eval:.1f} | Time: {elapsed/60:.1f}m")
            
            if best_eval >= 195:
                break
    
    env.close()

if __name__ == "__main__":
    main()