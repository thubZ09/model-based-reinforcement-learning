import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0

class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int, action_low, action_high):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)
        self.register_buffer("action_scale", torch.as_tensor((action_high - action_low) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias", torch.as_tensor((action_high + action_low) / 2.0, dtype=torch.float32))

    def forward(self, obs: torch.Tensor):
        h = self.net(obs)
        mu = self.mu(h)
        log_std = torch.clamp(self.log_std(h), LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

def watch_agent():
    device = torch.device("cpu")     
    env = gym.make("HalfCheetah-v5", render_mode="human")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    action_low = env.action_space.low
    action_high = env.action_space.high
    sac_hidden = 256 
    actor = Actor(obs_dim, act_dim, sac_hidden, action_low, action_high).to(device)
    actor.load_state_dict(torch.load("mbpo_halfcheetah_actor.pth", map_location=device, weights_only=True))
    actor.eval() 
    
    for episode in range(3): 
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            o = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                mu, _ = actor(o)
                a = (torch.tanh(mu) * actor.action_scale + actor.action_bias).cpu().numpy()[0]
            obs, reward, term, trunc, _ = env.step(a)
            total_reward += reward
            done = term or trunc
        print(f"episode {episode + 1} finished with reward: {total_reward:.1f}")
    env.close()

if __name__ == "__main__":
    watch_agent()