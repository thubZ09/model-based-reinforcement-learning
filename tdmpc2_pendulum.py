from __future__ import annotations
import random
import time
from dataclasses import dataclass
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro
from torch.distributions import Normal

@dataclass
class Args:
    exp_name: str = "tdmpc2_pendulum"
    env_id: str = "Pendulum-v1"
    seed: int = 1
    total_steps: int = 30_000
    init_random_steps: int = 1_000
    eval_every: int = 1_000
    eval_episodes: int = 5
    track: bool = False
    wandb_project: str = "cleanmbrl"
    latent_dim: int = 50
    simnorm_groups: int = 5
    hidden: int = 256
    mlp_layers: int = 2
    num_q: int = 5
    num_bins: int = 101                  
    vmin: float = -10.0                 
    vmax: float = 10.0
    lr: float = 3e-4
    weight_decay: float = 1e-5
    batch_size: int = 256
    horizon_train: int = 5             
    gamma: float = 0.99
    tau: float = 0.01
    grad_clip: float = 20.0
    train_every: int = 1
    updates_per_step: int = 1
    rho: float = 0.5                    
    plan_horizon: int = 5
    num_samples: int = 512
    num_elites: int = 64
    num_pi_samples: int = 24            
    plan_iterations: int = 6
    plan_temperature: float = 0.5
    plan_min_std: float = 0.05
    plan_max_std: float = 2.0
    buffer_size: int = 1_000_000

def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def set_seed(s): random.seed(s); np.random.seed(s); torch.manual_seed(s)
def soft_update(target, source, tau):
    with torch.no_grad():
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.mul_(1 - tau).add_(tau * s.data)

def two_hot(x: torch.Tensor, vmin: float, vmax: float, num_bins: int) -> torch.Tensor:
    x = x.clamp(vmin, vmax)
    bins = torch.linspace(vmin, vmax, num_bins, device=x.device)
    idx = ((x - vmin) / (vmax - vmin) * (num_bins - 1))
    lo = idx.floor().long().clamp(0, num_bins - 1)
    hi = (lo + 1).clamp(max=num_bins - 1)
    w_hi = (idx - lo.float())
    w_lo = 1.0 - w_hi
    out = torch.zeros(*x.shape, num_bins, device=x.device)
    out.scatter_(-1, lo.unsqueeze(-1), w_lo.unsqueeze(-1))
    out.scatter_(-1, hi.unsqueeze(-1), w_hi.unsqueeze(-1))
    return out

def two_hot_inv(logits: torch.Tensor, vmin: float, vmax: float, num_bins: int) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    bins = torch.linspace(vmin, vmax, num_bins, device=logits.device)
    return (probs * bins).sum(-1, keepdim=True)

def simnorm(z: torch.Tensor, num_groups: int) -> torch.Tensor:
    shape = z.shape
    z = z.view(*shape[:-1], num_groups, shape[-1] // num_groups)
    z = F.softmax(z, dim=-1)
    return z.view(*shape)

def mlp(in_dim, out_dim, hidden, layers, last_relu=False):
    mods = [nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.Mish()]
    for _ in range(layers - 1):
        mods += [nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.Mish()]
    mods.append(nn.Linear(hidden, out_dim))
    if last_relu:
        mods.append(nn.Mish())
    return nn.Sequential(*mods)

class WorldModel(nn.Module):
    def __init__(self, obs_dim, act_dim, args: Args, action_low, action_high):
        super().__init__()
        self.args = args
        self.act_dim = act_dim
        self.encoder = mlp(obs_dim, args.latent_dim, args.hidden, args.mlp_layers, last_relu=False)
        self.dynamics = mlp(args.latent_dim + act_dim, args.latent_dim, args.hidden, args.mlp_layers)
        self.reward = mlp(args.latent_dim + act_dim, args.num_bins, args.hidden, args.mlp_layers)
        self.pi_trunk = mlp(args.latent_dim, args.hidden, args.hidden, args.mlp_layers, last_relu=True)
        self.pi_mu = nn.Linear(args.hidden, act_dim)
        self.pi_log_std = nn.Linear(args.hidden, act_dim)
        self.qs = nn.ModuleList([
            mlp(args.latent_dim + act_dim, args.num_bins, args.hidden, args.mlp_layers)
            for _ in range(args.num_q)
        ])
        self.register_buffer("scale", torch.as_tensor((action_high - action_low) / 2.0, dtype=torch.float32))
        self.register_buffer("bias",  torch.as_tensor((action_high + action_low) / 2.0, dtype=torch.float32))
    def encode(self, obs):
        z = self.encoder(obs)
        return simnorm(z, self.args.simnorm_groups)
    def next_latent(self, z, a):
        return simnorm(self.dynamics(torch.cat([z, a], -1)), self.args.simnorm_groups)
    def reward_logits(self, z, a):
        return self.reward(torch.cat([z, a], -1))
    def q_logits(self, z, a):
        return [q(torch.cat([z, a], -1)) for q in self.qs]
    def pi(self, z, deterministic=False):
        h = self.pi_trunk(z)
        mu = self.pi_mu(h)
        log_std = self.pi_log_std(h).clamp(-5, 2)
        if deterministic:
            x = mu
            y = torch.tanh(x)
            return y * self.scale + self.bias, None
        std = log_std.exp()
        dist = Normal(mu, std)
        x = dist.rsample()
        y = torch.tanh(x)
        action = y * self.scale + self.bias
        logp = (dist.log_prob(x) - torch.log(self.scale * (1 - y.pow(2)) + 1e-6)).sum(-1, keepdim=True)
        return action, logp

class ReplayBuffer:
    def __init__(self, cap, obs_dim, act_dim, device):
        self.cap = int(cap); self.size = 0; self.idx = 0; self.device = device
        self.obs = np.zeros((self.cap, obs_dim), dtype=np.float32)
        self.act = np.zeros((self.cap, act_dim), dtype=np.float32)
        self.rew = np.zeros((self.cap, 1), dtype=np.float32)
        self.next_obs = np.zeros((self.cap, obs_dim), dtype=np.float32)
        self.done = np.zeros((self.cap, 1), dtype=np.float32)
    def add(self, o, a, r, no, d):
        i = self.idx
        self.obs[i] = o; self.act[i] = a; self.rew[i] = r
        self.next_obs[i] = no; self.done[i] = float(d)
        self.idx = (self.idx + 1) % self.cap
        self.size = min(self.size + 1, self.cap)
    def sample_seq(self, batch_size, horizon):
        starts = np.random.randint(0, self.size - horizon - 1, size=batch_size)
        obs = np.stack([self.obs[s : s + horizon + 1] for s in starts])
        act = np.stack([self.act[s : s + horizon + 1] for s in starts])
        rew = np.stack([self.rew[s : s + horizon + 1] for s in starts])
        done = np.stack([self.done[s : s + horizon + 1] for s in starts])
        next_obs = np.stack([self.next_obs[s : s + horizon + 1] for s in starts])
        return (torch.as_tensor(obs, device=self.device),
                torch.as_tensor(act, device=self.device),
                torch.as_tensor(rew, device=self.device),
                torch.as_tensor(next_obs, device=self.device),
                torch.as_tensor(done, device=self.device))

@torch.no_grad()
def plan(model: WorldModel, obs: np.ndarray, args: Args, device, prev_mean=None):
    z0 = model.encode(torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))
    z0 = z0.expand(args.num_samples + args.num_pi_samples, -1)
    H, A = args.plan_horizon, model.act_dim
    if prev_mean is None:
        mean = torch.zeros(H, A, device=device)
    else:
        mean = prev_mean
        mean = torch.cat([mean[1:], torch.zeros(1, A, device=device)], dim=0)
    std = torch.full((H, A), args.plan_max_std, device=device)
    for it in range(args.plan_iterations):
        eps = torch.randn(args.num_samples, H, A, device=device)
        sampled = (mean.unsqueeze(0) + std.unsqueeze(0) * eps)
        sampled = (torch.tanh(sampled) * model.scale + model.bias)
        pi_actions = torch.zeros(args.num_pi_samples, H, A, device=device)
        z = z0[: args.num_pi_samples]
        for t in range(H):
            a, _ = model.pi(z)
            pi_actions[:, t] = a
            z = model.next_latent(z, _norm_action(a, model))
        all_actions = torch.cat([sampled, pi_actions], dim=0)      
        z = z0.clone()
        ret = torch.zeros(all_actions.shape[0], device=device)
        for t in range(H):
            a = all_actions[:, t]
            a_norm = _norm_action(a, model)
            r = two_hot_inv(model.reward_logits(z, a_norm), -abs(args.vmin), abs(args.vmax), args.num_bins).squeeze(-1)
            ret = ret + (args.gamma ** t) * r
            z = model.next_latent(z, a_norm)
        a_end, _ = model.pi(z)
        a_end_norm = _norm_action(a_end, model)
        q_logits_list = model.q_logits(z, a_end_norm)
        qs = torch.stack([two_hot_inv(l, args.vmin, args.vmax, args.num_bins).squeeze(-1) for l in q_logits_list], 0)
        q_min = qs.min(0).values
        ret = ret + (args.gamma ** H) * q_min
        top_idx = ret.topk(args.num_elites).indices
        top_actions = all_actions[top_idx]                        
        score = ret[top_idx]
        weights = F.softmax((score - score.max()) / args.plan_temperature, dim=0)
        unscaled = (top_actions - model.bias) / model.scale
        unscaled = torch.atanh(unscaled.clamp(-0.999, 0.999))
        mean = (weights.view(-1, 1, 1) * unscaled).sum(0)
        var = (weights.view(-1, 1, 1) * (unscaled - mean.unsqueeze(0)).pow(2)).sum(0)
        std = var.sqrt().clamp(args.plan_min_std, args.plan_max_std)
    final = mean[0] + std[0] * torch.randn_like(mean[0])
    final_action = (torch.tanh(final) * model.scale + model.bias).cpu().numpy()
    return final_action, mean

def _norm_action(a, model):
    return (a - model.bias) / model.scale
def train_step(model: WorldModel, target_model: WorldModel, opt, buf: ReplayBuffer, args: Args, device):
    obs, act, rew, next_obs, done = buf.sample_seq(args.batch_size, args.horizon_train)
    z = model.encode(obs[:, 0])
    total_loss = 0.0
    dyn_loss_acc = 0.0; rew_loss_acc = 0.0; q_loss_acc = 0.0; pi_loss_acc = 0.0
    for t in range(args.horizon_train):
        a_t = _norm_action(act[:, t], model)
        z_next_pred = model.next_latent(z, a_t)
        with torch.no_grad():
            z_next_target = target_model.encode(next_obs[:, t])
        dyn_loss = F.mse_loss(z_next_pred, z_next_target)
        r_logits = model.reward_logits(z, a_t)
        r_target = two_hot(rew[:, t].squeeze(-1), -abs(args.vmin), abs(args.vmax), args.num_bins)
        rew_loss = -(r_target * F.log_softmax(r_logits, dim=-1)).sum(-1).mean()
        with torch.no_grad():
            next_a, _ = target_model.pi(z_next_target)
            next_a_n = _norm_action(next_a, target_model)
            tq_logits = target_model.q_logits(z_next_target, next_a_n)
            tq = torch.stack([two_hot_inv(l, args.vmin, args.vmax, args.num_bins).squeeze(-1) for l in tq_logits], 0)
            tq_min = tq.min(0).values
            td_target = rew[:, t].squeeze(-1) + args.gamma * (1.0 - done[:, t].squeeze(-1)) * tq_min
        q_logits_list = model.q_logits(z, a_t)
        q_target_two_hot = two_hot(td_target, args.vmin, args.vmax, args.num_bins)
        q_loss = sum(-(q_target_two_hot * F.log_softmax(q, dim=-1)).sum(-1).mean() for q in q_logits_list) / args.num_q
        w = args.rho ** t
        total_loss = total_loss + w * (dyn_loss + rew_loss + q_loss)
        dyn_loss_acc += dyn_loss.item(); rew_loss_acc += rew_loss.item(); q_loss_acc += q_loss.item()
        z = z_next_pred
    z0 = model.encode(obs[:, 0]).detach()
    a_pi, logp = model.pi(z0)
    a_pi_norm = _norm_action(a_pi, model)
    q_pi = torch.stack([
        two_hot_inv(l, args.vmin, args.vmax, args.num_bins).squeeze(-1)
        for l in model.q_logits(z0, a_pi_norm)
    ], 0).min(0).values
    pi_loss = -q_pi.mean() + 0.01 * logp.mean()
    pi_loss_acc = pi_loss.item()

    loss = total_loss + pi_loss
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    opt.step()
    soft_update(target_model, model, args.tau)
    return {"loss/dyn": dyn_loss_acc, "loss/rew": rew_loss_acc, "loss/q": q_loss_acc, "loss/pi": pi_loss_acc}

@torch.no_grad()
def evaluate(env, model, args, device, n):
    rets = []
    for _ in range(n):
        obs, _ = env.reset(); ret, done = 0.0, False; prev = None
        while not done:
            a, prev = plan(model, obs, args, device, prev_mean=prev)
            obs, r, term, trunc, _ = env.step(a); ret += r; done = term or trunc
        rets.append(ret)
    return float(np.mean(rets))

def main(args: Args):
    set_seed(args.seed)
    device = get_device()
    print(f"[tdmpc2] device={device}")
    env = gym.make(args.env_id); eval_env = gym.make(args.env_id)
    env.action_space.seed(args.seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    model = WorldModel(obs_dim, act_dim, args, env.action_space.low, env.action_space.high).to(device)
    target_model = WorldModel(obs_dim, act_dim, args, env.action_space.low, env.action_space.high).to(device)
    target_model.load_state_dict(model.state_dict())
    for p in target_model.parameters(): p.requires_grad_(False)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    buf = ReplayBuffer(args.buffer_size, obs_dim, act_dim, device)
    if args.track:
        import wandb; wandb.init(project=args.wandb_project, name=args.exp_name, config=vars(args))
    obs, _ = env.reset(seed=args.seed)
    prev_plan = None; best = -float("inf"); start = time.time()
    for step in range(1, args.total_steps + 1):
        if step < args.init_random_steps or buf.size < args.batch_size + args.horizon_train:
            a = env.action_space.sample(); prev_plan = None
        else:
            a, prev_plan = plan(model, obs, args, device, prev_mean=prev_plan)
        no, r, term, trunc, _ = env.step(a)
        buf.add(obs, a, r, no, term)
        if term or trunc:
            obs, _ = env.reset(); prev_plan = None
        else:
            obs = no
        if step >= args.init_random_steps and buf.size > args.batch_size + args.horizon_train and step % args.train_every == 0:
            for _ in range(args.updates_per_step):
                stats = train_step(model, target_model, opt, buf, args, device)
            if args.track and step % 500 == 0:
                wandb.log(stats, step=step)
        if step % args.eval_every == 0 and step >= args.init_random_steps:
            ret = evaluate(eval_env, model, args, device, args.eval_episodes)
            best = max(best, ret)
            print(f"step={step:6d}  eval={ret:8.1f}  best={best:8.1f}  buf={buf.size}  "
                  f"elapsed={(time.time()-start)/60:5.1f}m")
            if args.track: wandb.log({"eval/return": ret, "eval/best": best}, step=step)
    print(f"[tdmpc2] done. best={best:.2f}")

if __name__ == "__main__":
    main(tyro.cli(Args))