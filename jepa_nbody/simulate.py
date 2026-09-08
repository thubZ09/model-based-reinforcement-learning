from __future__ import annotations
import torch
G = 1.0  

def accelerations(pos: torch.Tensor, mass: torch.Tensor, softening: float = 1e-2) -> torch.Tensor:
    diff = pos.unsqueeze(-2) - pos.unsqueeze(-3)          
    dist2 = (diff ** 2).sum(-1) + softening ** 2        
    inv_dist3 = dist2.pow(-1.5)
    n = pos.shape[-2]
    eye = torch.eye(n, device=pos.device, dtype=pos.dtype)
    inv_dist3 = inv_dist3 * (1.0 - eye)                   
    m_j = mass.unsqueeze(-2)                             
    acc = -G * (diff * inv_dist3.unsqueeze(-1) * m_j.unsqueeze(-1)).sum(-2)
    return acc

def velocity_verlet_step(pos, vel, mass, dt: float, softening: float = 1e-2):
    acc = accelerations(pos, mass, softening)
    pos_new = pos + vel * dt + 0.5 * acc * dt ** 2
    acc_new = accelerations(pos_new, mass, softening)
    vel_new = vel + 0.5 * (acc + acc_new) * dt
    return pos_new, vel_new

def total_energy(pos: torch.Tensor, vel: torch.Tensor, mass: torch.Tensor, softening: float = 1e-2) -> torch.Tensor:
    extra_dims = pos.dim() - mass.dim() - 1
    m = mass
    for _ in range(extra_dims):
        m = m.unsqueeze(-2)
    ke = 0.5 * (m * (vel ** 2).sum(-1)).sum(-1)
    diff = pos.unsqueeze(-2) - pos.unsqueeze(-3)
    dist = (diff ** 2).sum(-1).add(softening ** 2).sqrt()
    n = pos.shape[-2]
    eye = torch.eye(n, device=pos.device, dtype=pos.dtype)
    inv_dist = (1.0 / dist) * (1.0 - eye)
    m_i = m.unsqueeze(-1)
    m_j = m.unsqueeze(-2)
    pe = -G * 0.5 * (m_i * m_j * inv_dist).sum(-1).sum(-1) 
    return ke + pe

@torch.no_grad()
def generate(n_traj: int, n_frames: int, n_bodies: int = 4, dt: float = 0.01,
             substeps: int = 20, seed: int = 0, device: str = "cpu",
             softening: float = 0.08):
    torch.manual_seed(seed)
    mass = torch.rand(n_traj, n_bodies, device=device) * 0.9 + 0.1
    mass[:, 0] = mass[:, 0] * 5 + 5  
    pos = torch.zeros(n_traj, n_bodies, 2, device=device)
    vel = torch.zeros(n_traj, n_bodies, 2, device=device)
    n_planets = n_bodies - 1
    base_radius = 1.0 + 0.8 * torch.arange(n_planets, device=device).float()
    radius = base_radius.unsqueeze(0) + torch.rand(n_traj, n_planets, device=device) * 0.3
    angle = torch.rand(n_traj, n_planets, device=device) * 2 * 3.14159265
    pos[:, 1:, 0] = radius * torch.cos(angle)
    pos[:, 1:, 1] = radius * torch.sin(angle)
    v_mag = torch.sqrt(G * mass[:, 0:1] / radius.clamp(min=0.5))
    jitter = 0.9 + 0.15 * torch.rand(n_traj, n_planets, device=device)  
    vel[:, 1:, 0] = -v_mag * torch.sin(angle) * jitter
    vel[:, 1:, 1] = v_mag * torch.cos(angle) * jitter
    pos += 0.02 * torch.randn_like(pos)
    vel += 0.01 * torch.randn_like(vel)
    frames_pos, frames_vel = [], []
    for _ in range(n_frames):
        for _ in range(substeps):
            pos, vel = velocity_verlet_step(pos, vel, mass, dt, softening)
        frames_pos.append(pos.clone())
        frames_vel.append(vel.clone())
    traj_pos = torch.stack(frames_pos, dim=1)
    traj_vel = torch.stack(frames_vel, dim=1)
    return traj_pos, traj_vel, mass