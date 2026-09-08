from __future__ import annotations
import os
import time
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F
from model import JEPANBody, variance_reg
from simulate import generate, total_energy

@dataclass
class Args:
    n_bodies: int = 4
    n_train_traj: int = 256
    n_test_traj: int = 32
    n_frames: int = 60
    dt: float = 0.01
    substeps: int = 20
    latent_dim: int = 64
    horizon: int = 10
    batch_size: int = 32
    steps: int = 3000
    lr: float = 3e-4
    ema_tau: float = 0.99
    var_weight: float = 1.0
    dec_weight: float = 1.0
    latent_weight: float = 1.0
    eval_every: int = 300
    save_viz: bool = True
    seed: int = 1

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def to_state(pos: torch.Tensor, vel: torch.Tensor) -> torch.Tensor:
    return torch.cat([pos.flatten(-2), vel.flatten(-2)], dim=-1)

def from_state(state: torch.Tensor, n_bodies: int):
    pos_flat, vel_flat = state[..., : 2 * n_bodies], state[..., 2 * n_bodies:]
    pos = pos_flat.reshape(*state.shape[:-1], n_bodies, 2)
    vel = vel_flat.reshape(*state.shape[:-1], n_bodies, 2)
    return pos, vel

def sample_batch(states, mass, batch_size, horizon, device):
    N, T = states.shape[0], states.shape[1]
    idx = torch.randint(0, N, (batch_size,))
    start = torch.randint(0, T - horizon, (batch_size,))
    seqs = torch.stack([states[idx[b], start[b]:start[b] + horizon + 1] for b in range(batch_size)])
    return seqs.to(device), mass[idx].to(device)

@torch.no_grad()
def evaluate(model, states, mass, n_bodies, args, device):
    H = args.horizon
    seqs = states[:, :H + 1].to(device)
    m = mass.to(device)
    z = model.encoder(seqs[:, 0])
    field_errs, energy_drifts = [], []
    pos0, vel0 = from_state(seqs[:, 0], n_bodies)
    e0 = total_energy(pos0, vel0, m)
    for t in range(1, H + 1):
        z = model.predictor(z, m)
        pred_state = model.decoder(z)
        field_errs.append(F.mse_loss(pred_state, seqs[:, t]).item())
        pos_t, vel_t = from_state(pred_state, n_bodies)
        e_t = total_energy(pos_t, vel_t, m)
        energy_drifts.append(((e_t - e0) / e0.abs().clamp(min=1e-3)).abs().mean().item())
    return sum(field_errs) / H, sum(energy_drifts) / H

@torch.no_grad()
def measure_speedup(model, n_bodies, args, device):
    from simulate import velocity_verlet_step
    H = args.horizon
    mass = torch.rand(1, n_bodies, device=device) * 0.9 + 0.1
    mass[:, 0] = mass[:, 0] * 5 + 5
    pos = torch.randn(1, n_bodies, 2, device=device) * 0.5
    vel = torch.randn(1, n_bodies, 2, device=device) * 0.2
    t0 = time.time()
    p, v = pos.clone(), vel.clone()
    for _ in range(H * args.substeps):
        p, v = velocity_verlet_step(p, v, mass, args.dt)
    solver_t = time.time() - t0
    z = model.encoder(to_state(pos, vel))
    t0 = time.time()
    for _ in range(H):
        z = model.predictor(z, mass)
    latent_t = time.time() - t0
    return solver_t / max(latent_t, 1e-6)

def save_viz(model, states, mass, n_bodies, args, device, path="results/jepa_nbody_orbits.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    H = min(30, args.n_frames - 1)
    seq = states[:1, :H + 1].to(device)
    m = mass[:1].to(device)
    z = model.encoder(seq[:, 0])
    true_pos, _ = from_state(seq[0], n_bodies)    
    pred_positions = [true_pos[0].cpu().numpy()]
    with torch.no_grad():
        for t in range(1, H + 1):
            z = model.predictor(z, m)
            pred_state = model.decoder(z)
            pred_pos, _ = from_state(pred_state[0], n_bodies)
            pred_positions.append(pred_pos.cpu().numpy())
    pred_positions = np.stack(pred_positions)      
    true_positions = true_pos.cpu().numpy()
    colors = plt.cm.tab10(np.linspace(0, 1, n_bodies))
    fig, ax = plt.subplots(figsize=(7, 7), dpi=140)
    for b in range(n_bodies):
        ax.plot(true_positions[:, b, 0], true_positions[:, b, 1],
                color=colors[b], linewidth=2.2, label=f"body {b} (true)")
        ax.plot(pred_positions[:, b, 0], pred_positions[:, b, 1],
                color=colors[b], linewidth=2.2, linestyle="--", alpha=0.7,
                label=f"body {b} (JEPA latent rollout)")
        ax.scatter(*true_positions[0, b], color=colors[b], s=60, zorder=5, edgecolor="black")
    ax.set_title(f"N-body orbits: true vs {H}-step JEPA latent rollout", fontsize=12, fontweight="bold")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.legend(fontsize=7, ncol=2, loc="upper right", frameon=False)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"[jepa-nbody] saved {path}", flush=True)

def main(args: Args):
    torch.manual_seed(args.seed)
    device = get_device()
    print(f"[jepa-nbody] device={device}  generating data...", flush=True)
    train_pos, train_vel, train_mass = generate(
        args.n_train_traj, args.n_frames, args.n_bodies, args.dt, args.substeps, seed=args.seed
    )
    test_pos, test_vel, test_mass = generate(
        args.n_test_traj, args.n_frames, args.n_bodies, args.dt, args.substeps, seed=args.seed + 999
    )
    train_states = to_state(train_pos, train_vel)
    test_states = to_state(test_pos, test_vel)
    state_dim = train_states.shape[-1]
    print(f"[jepa-nbody] train {tuple(train_states.shape)}  test {tuple(test_states.shape)}", flush=True)
    model = JEPANBody(state_dim, args.n_bodies, args.latent_dim, args.ema_tau).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    start = time.time()
    for step in range(1, args.steps + 1):
        seqs, m = sample_batch(train_states, train_mass, args.batch_size, args.horizon, device)
        z = model.encoder(seqs[:, 0])
        pred_loss = 0.0
        var_loss = 0.0
        rec_loss = F.smooth_l1_loss(model.decoder(z), seqs[:, 0])
        for t in range(1, args.horizon + 1):
            z = model.predictor(z, m)
            with torch.no_grad():
                target = model.target_encoder(seqs[:, t])
            pred_loss = pred_loss + F.smooth_l1_loss(z, target)
            rec_loss = rec_loss + F.smooth_l1_loss(model.decoder(z), seqs[:, t])
            var_loss = var_loss + variance_reg(z)
        pred_loss /= args.horizon
        var_loss /= args.horizon
        rec_loss /= (args.horizon + 1)
        loss = args.dec_weight * rec_loss + args.latent_weight * pred_loss + args.var_weight * var_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.update_target()
        if step % args.eval_every == 0:
            field_err, energy_drift = evaluate(model, test_states, test_mass, args.n_bodies, args, device)
            speed = measure_speedup(model, args.n_bodies, args, device)
            print(f"step={step:5d}  rec={rec_loss.item():.4f}  pred={pred_loss.item():.4f}  "
                  f"var={var_loss.item():.4f}  field_err={field_err:.4f}  "
                  f"energy_drift={energy_drift:.3f}  speedup={speed:.1f}x  "
                  f"t={(time.time()-start)/60:.1f}m", flush=True)
    if args.save_viz:
        save_viz(model, test_states, test_mass, args.n_bodies, args, device)
    print("[jepa-nbody] done", flush=True)

if __name__ == "__main__":
    import tyro
    main(tyro.cli(Args))