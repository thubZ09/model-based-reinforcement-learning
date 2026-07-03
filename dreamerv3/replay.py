from __future__ import annotations
import random
from dataclasses import dataclass, field
import numpy as np
import torch

@dataclass
class Episode:
    obs: list = field(default_factory=list)     
    action: list = field(default_factory=list)
    reward: list = field(default_factory=list)
    cont: list = field(default_factory=list)  
    is_first: list = field(default_factory=list)
    def __len__(self):
        return len(self.action)

class EpisodicReplay:
    def __init__(self, capacity_steps: int, seq_len: int):
        self.capacity = capacity_steps
        self.seq_len = seq_len
        self.episodes: list[Episode] = []
        self.total = 0
    def add_episode(self, ep: Episode):
        self.episodes.append(ep)
        self.total += len(ep)
        while self.total > self.capacity and len(self.episodes) > 1:
            removed = self.episodes.pop(0)
            self.total -= len(removed)
    def ready(self) -> bool:
        return any(len(e) >= self.seq_len for e in self.episodes)
    def sample(self, batch_size: int, device: torch.device) -> dict:
        valid = [e for e in self.episodes if len(e) >= self.seq_len]
        eps = random.choices(valid, k=batch_size)
        obs_b, act_b, rew_b, cont_b, first_b = [], [], [], [], []
        for ep in eps:
            start = random.randint(0, len(ep) - self.seq_len)
            sl = slice(start, start + self.seq_len)
            obs_b.append(np.stack(ep.obs[sl], 0))
            act_b.append(np.stack(ep.action[sl], 0))
            rew_b.append(np.asarray(ep.reward[sl], dtype=np.float32))
            cont_b.append(np.asarray(ep.cont[sl], dtype=np.float32))
            first_b.append(np.asarray(ep.is_first[sl], dtype=np.float32))
        obs = np.stack(obs_b, 0).astype(np.float32) / 255.0 - 0.5
        obs = np.transpose(obs, (0, 1, 4, 2, 3))
        return {
            "obs":      torch.as_tensor(obs, device=device),
            "action":   torch.as_tensor(np.stack(act_b, 0).astype(np.float32), device=device),
            "reward":   torch.as_tensor(np.stack(rew_b, 0), device=device),
            "cont":     torch.as_tensor(np.stack(cont_b, 0), device=device),
            "is_first": torch.as_tensor(np.stack(first_b, 0), device=device),
        }