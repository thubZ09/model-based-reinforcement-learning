from __future__ import annotations
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

def mlp(in_dim, out_dim, hidden, layers):
    mods = [nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.SiLU()]
    for _ in range(layers - 1):
        mods += [nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU()]
    mods.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*mods)

class Encoder(nn.Module):
    def __init__(self, state_dim: int, latent_dim: int, hidden: int = 128):
        super().__init__()
        self.net = mlp(state_dim, latent_dim, hidden, 2)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)

class Predictor(nn.Module):
    def __init__(self, latent_dim: int, n_bodies: int, hidden: int = 128):
        super().__init__()
        self.film = nn.Linear(n_bodies, 2 * latent_dim)
        self.trunk = mlp(latent_dim, latent_dim, hidden, 2)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, z: torch.Tensor, mass: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.film(mass).chunk(2, dim=-1)
        h = self.norm(z) * (1 + gamma) + beta
        delta = self.trunk(h)
        return z + delta  

class Decoder(nn.Module):
    def __init__(self, latent_dim: int, state_dim: int, hidden: int = 128):
        super().__init__()
        self.net = mlp(latent_dim, state_dim, hidden, 2)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

class JEPANBody(nn.Module):
    def __init__(self, state_dim: int, n_bodies: int, latent_dim: int = 64, ema_tau: float = 0.99):
        super().__init__()
        self.encoder = Encoder(state_dim, latent_dim)
        self.predictor = Predictor(latent_dim, n_bodies)
        self.decoder = Decoder(latent_dim, state_dim)
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        self.ema_tau = ema_tau
    @torch.no_grad()
    def update_target(self):
        for t, s in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            t.data.mul_(self.ema_tau).add_((1 - self.ema_tau) * s.data)

def variance_reg(z: torch.Tensor) -> torch.Tensor:
    z = z.flatten(0, -2) if z.dim() > 2 else z
    std = torch.sqrt(z.var(dim=0) + 1e-4)
    return torch.relu(1.0 - std).mean()