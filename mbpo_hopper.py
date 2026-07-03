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
    exp_name: str = "mbpo_hopper"
    env_id: str = "Hopper-v5"
    seed: int = 1
    total_steps: int = 300_000
    eval_every: int = 5_000
    eval_episodes: int = 5
    track: bool = False
    wandb_project: str = "cleanmbrl"
    sac_lr: float = 3e-4
    sac_gamma: float = 0.99
    sac_tau: float = 0.005
    sac_batch_size: int = 256
    sac_updates_per_step: int = 40           
    sac_hidden: int = 256
    real_ratio: float = 0.05
    model_lr: float = 1e-3
    model_weight_decay: float = 1e-5
    ensemble_size: int = 7
    elite_size: int = 5
    model_hidden: int = 200
    model_layers: int = 4
    model_batch_size: int = 256
    model_train_freq: int = 250
    model_max_epochs: int = 50
    model_max_epochs_since_update: int = 5
    model_val_split: float = 0.2
    rollout_freq: int = 250
    rollout_batch_size: int = 50_000
    rollout_length_start: int = 1
    rollout_length_end: int = 15
    rollout_length_warmup: int = 20_000
    rollout_length_anneal: int = 100_000
    init_random_steps: int = 5_000
    env_buffer_size: int = 1_000_000
    model_buffer_size: int = 400_000
    term_threshold: float = 0.5

def get_device() -> torch.device:
    return torch.device("cpu")

def configure_device(device):
    if device.type == "mps":
        torch.set_default_dtype(torch.float32)
def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

def soft_update(target, source, tau):
    with torch.no_grad():
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.mul_(1 - tau).add_(tau * s.data)

class ReplayBuffer:
    def __init__(self, cap, obs_dim, act_dim, device):
        self.cap = int(cap); self.idx = 0; self.size = 0; self.device = device
        self.obs = np.zeros((self.cap, obs_dim), dtype=np.float32)
        self.act = np.zeros((self.cap, act_dim), dtype=np.float32)
        self.rew = np.zeros((self.cap, 1), dtype=np.float32)
        self.next_obs = np.zeros((self.cap, obs_dim), dtype=np.float32)
        self.done = np.zeros((self.cap, 1), dtype=np.float32)

    def add(self, o, a, r, no, d):
        i = self.idx
        self.obs[i] = o; self.act[i] = a; self.rew[i] = r
        self.next_obs[i] = no; self.done[i] = float(d) if np.ndim(d) == 0 else float(np.squeeze(d))
        self.idx = (self.idx + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def add_batch(self, obs, act, rew, next_obs, done):
        n = obs.shape[0]
        indices = np.arange(self.idx, self.idx + n) % self.cap
        self.obs[indices] = obs
        self.act[indices] = act
        self.rew[indices] = rew.reshape(n, 1) if rew.ndim == 1 else rew
        self.next_obs[indices] = next_obs
        self.done[indices] = done.reshape(n, 1) if done.ndim == 1 else done
        self.idx = (self.idx + n) % self.cap
        self.size = min(self.size + n, self.cap)

    def sample(self, bs):
        idx = np.random.randint(0, self.size, size=bs)
        return tuple(torch.as_tensor(arr[idx], device=self.device) for arr in
                     (self.obs, self.act, self.rew, self.next_obs, self.done))

    def sample_states(self, bs):
        idx = np.random.randint(0, self.size, size=bs)
        return self.obs[idx].copy()

class EnsembleLinear(nn.Module):
    def __init__(self, in_dim, out_dim, E):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(E, in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(E, 1, out_dim))
        for i in range(E):
            nn.init.trunc_normal_(self.weight[i], std=1.0 / (2.0 * np.sqrt(in_dim)))

    def forward(self, x):
        return torch.bmm(x, self.weight) + self.bias

class GaussianMLPEnsemble(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden, layers, E):
        super().__init__()
        self.obs_dim = obs_dim; self.act_dim = act_dim; self.E = E
        self.output_dim = obs_dim + 1 + 1                        
        in_dim = obs_dim + act_dim
        self.layers = nn.ModuleList()
        for i in range(layers):
            self.layers.append(EnsembleLinear(in_dim if i == 0 else hidden, hidden, E))
        self.mean_head = EnsembleLinear(hidden, self.output_dim, E)
        self.logvar_head = EnsembleLinear(hidden, obs_dim + 1, E) 
        self.max_logvar = nn.Parameter(torch.full((1, 1, obs_dim + 1), 0.5))
        self.min_logvar = nn.Parameter(torch.full((1, 1, obs_dim + 1), -10.0))
        self.register_buffer("in_mean", torch.zeros(in_dim))
        self.register_buffer("in_std", torch.ones(in_dim))

    def forward(self, obs, act):
        if obs.ndim == 2:
            obs = obs.unsqueeze(0).expand(self.E, -1, -1)
            act = act.unsqueeze(0).expand(self.E, -1, -1)
        x = torch.cat([obs, act], -1)
        x = (x - self.in_mean) / (self.in_std + 1e-6)
        for layer in self.layers:
            x = F.silu(layer(x))
        mean = self.mean_head(x)
        gaussian_mean = mean[..., : self.obs_dim + 1]
        term_logit = mean[..., -1:]
        logvar = self.logvar_head(x)
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        return gaussian_mean, logvar, term_logit

    def update_norm(self, obs_np, act_np):
        x = np.concatenate([obs_np, act_np], -1)
        self.in_mean.copy_(torch.as_tensor(x.mean(0), dtype=torch.float32))
        self.in_std.copy_(torch.as_tensor(x.std(0) + 1e-6, dtype=torch.float32))

def train_dynamics(model, optim, buf: ReplayBuffer, args, device):
    N = buf.size
    obs = buf.obs[:N].copy(); act = buf.act[:N].copy()
    rew = buf.rew[:N].copy(); next_obs = buf.next_obs[:N].copy(); done = buf.done[:N].copy()
    delta = next_obs - obs
    gaussian_target = np.concatenate([delta, rew], -1)
    term_target = done
    model.update_norm(obs, act)
    perm = np.random.permutation(N)
    n_val = int(N * args.model_val_split)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    obs_t = torch.as_tensor(obs, device=device); act_t = torch.as_tensor(act, device=device)
    gtgt_t = torch.as_tensor(gaussian_target, device=device)
    term_t = torch.as_tensor(term_target, device=device)
    val = (obs_t[val_idx], act_t[val_idx], gtgt_t[val_idx], term_t[val_idx])
    tr_obs, tr_act, tr_gtgt, tr_term = obs_t[tr_idx], act_t[tr_idx], gtgt_t[tr_idx], term_t[tr_idx]
    n = len(tr_idx)
    boot = np.stack([np.random.randint(0, n, n) for _ in range(model.E)])
    best = [float("inf")] * model.E
    stale = 0
    bs = args.model_batch_size
    for epoch in range(args.model_max_epochs):
        for i in range(model.E):
            np.random.shuffle(boot[i])
        for start in range(0, n, bs):
            end = min(start + bs, n)
            idx = torch.as_tensor(boot[:, start:end], device=device, dtype=torch.long)
            o = tr_obs[idx]; a = tr_act[idx]
            y_gauss = tr_gtgt[idx]; y_term = tr_term[idx]
            mean, logvar, term_logit = model(o, a)
            inv_var = (-logvar).exp()
            nll = ((mean - y_gauss).pow(2) * inv_var + logvar).mean()
            bce = F.binary_cross_entropy_with_logits(term_logit, y_term)
            loss = nll + bce + 0.01 * (model.max_logvar.sum() - model.min_logvar.sum())
            optim.zero_grad(); loss.backward(); optim.step()
        with torch.no_grad():
            mean, _, _ = model(val[0], val[1])
            tgt = val[2].unsqueeze(0).expand_as(mean)
            val_mse = (mean - tgt).pow(2).mean(dim=(1, 2))
        improved = False
        for i in range(model.E):
            v = val_mse[i].item()
            if v < best[i] * 0.99:
                best[i] = v; improved = True
        stale = 0 if improved else stale + 1
        if stale >= args.model_max_epochs_since_update:
            break
    val_mse_final = val_mse.cpu().numpy()
    elite_inds = np.argsort(val_mse_final)[: args.elite_size].tolist()
    return {"model/val_mse": float(val_mse_final.mean()), "elite_inds": elite_inds, "model/epochs": epoch + 1}

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden, low, high):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU())
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)
        self.register_buffer("scale", torch.as_tensor((high - low) / 2.0, dtype=torch.float32))
        self.register_buffer("bias",  torch.as_tensor((high + low) / 2.0, dtype=torch.float32))
    def forward(self, obs):
        h = self.net(obs)
        mu = self.mu(h)
        log_std = torch.clamp(self.log_std(h), LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std
    def sample(self, obs):
        mu, log_std = self(obs)
        std = log_std.exp()
        normal = Normal(mu, std)
        x = normal.rsample()
        y = torch.tanh(x)
        action = y * self.scale + self.bias
        logp = (normal.log_prob(x) - torch.log(self.scale * (1 - y.pow(2)) + 1e-6)).sum(-1, keepdim=True)
        return action, logp

class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden):
        super().__init__()
        def build():
            return nn.Sequential(nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))
        self.q1 = build(); self.q2 = build()
    def forward(self, obs, act):
        x = torch.cat([obs, act], -1)
        return self.q1(x), self.q2(x)

@torch.no_grad()
def rollout_model(actor, model, elite_inds, env_buf, model_buf, k, n_start, args, device):
    starts = env_buf.sample_states(n_start)
    obs = torch.as_tensor(starts, device=device)
    alive_mask = torch.ones(n_start, dtype=torch.bool, device=device)
    for _ in range(k):
        if alive_mask.sum() == 0:
            break
        live_obs = obs[alive_mask]
        act, _ = actor.sample(live_obs)
        mean, logvar, term_logit = model(live_obs, act)
        std = (logvar * 0.5).exp()
        sample = mean + std * torch.randn_like(mean)
        choice = torch.as_tensor(np.random.choice(elite_inds, size=sample.shape[1]),
                                 device=device, dtype=torch.long)
        pred = sample[choice, torch.arange(sample.shape[1], device=device)]
        delta, reward = pred[:, :-1], pred[:, -1:]
        term_p = torch.sigmoid(term_logit).mean(0)                
        done = (term_p > args.term_threshold).float()
        next_obs = live_obs + delta
        model_buf.add_batch(
            live_obs.cpu().numpy(), act.cpu().numpy(),
            reward.cpu().numpy(), next_obs.cpu().numpy(), done.cpu().numpy(),
        )
        keep = (done.squeeze(-1) < 0.5)
        live_indices = alive_mask.nonzero(as_tuple=True)[0]
        died = live_indices[~keep]
        alive_mask = alive_mask.clone()
        alive_mask[died] = False
        new_obs = obs.clone()
        new_obs[live_indices] = next_obs
        obs = new_obs

def sac_update(actor, critic, target_critic, log_alpha, target_entropy,
               oa, oc, oalpha, env_buf, model_buf, args):
    bs = args.sac_batch_size
    n_real = int(bs * args.real_ratio)
    n_model = bs - n_real
    if model_buf.size > 0 and n_model > 0:
        if n_real > 0:
            ro, ra, rr, rno, rd = env_buf.sample(n_real)
            mo, ma, mr, mno, md = model_buf.sample(n_model)
            obs = torch.cat([ro, mo]); act = torch.cat([ra, ma])
            rew = torch.cat([rr, mr]); nobs = torch.cat([rno, mno]); done = torch.cat([rd, md])
        else:
            obs, act, rew, nobs, done = model_buf.sample(n_model)
    else:
        obs, act, rew, nobs, done = env_buf.sample(bs)
    alpha = log_alpha.exp().detach()
    with torch.no_grad():
        na, nlogp = actor.sample(nobs)
        q1t, q2t = target_critic(nobs, na)
        qt = torch.min(q1t, q2t) - alpha * nlogp
        y = rew + args.sac_gamma * (1 - done) * qt
    q1, q2 = critic(obs, act)
    closs = F.mse_loss(q1, y) + F.mse_loss(q2, y)
    oc.zero_grad(); closs.backward(); oc.step()
    a_new, logp = actor.sample(obs)
    q1n, q2n = critic(obs, a_new)
    aloss = (alpha * logp - torch.min(q1n, q2n)).mean()
    oa.zero_grad(); aloss.backward(); oa.step()
    alpha_loss = -(log_alpha * (logp.detach() + target_entropy)).mean()
    oalpha.zero_grad(); alpha_loss.backward(); oalpha.step()
    soft_update(target_critic, critic, args.sac_tau)

@torch.no_grad()
def evaluate(env, actor, device, n):
    rets = []
    for _ in range(n):
        obs, _ = env.reset(); ret, done = 0.0, False
        while not done:
            o = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            mu, _ = actor(o)
            a = (torch.tanh(mu) * actor.scale + actor.bias).cpu().numpy()[0]
            obs, r, term, trunc, _ = env.step(a); ret += r; done = term or trunc
        rets.append(ret)
    return float(np.mean(rets))

def current_rollout_length(step, args):
    if step < args.rollout_length_warmup: return args.rollout_length_start
    if step >= args.rollout_length_warmup + args.rollout_length_anneal: return args.rollout_length_end
    frac = (step - args.rollout_length_warmup) / args.rollout_length_anneal
    return int(round(args.rollout_length_start + frac * (args.rollout_length_end - args.rollout_length_start)))

def main(args: Args):
    set_seed(args.seed); device = get_device()
    configure_device(device)
    print(f"[mbpo-hopper] device={device}")
    env = gym.make(args.env_id); eval_env = gym.make(args.env_id)
    env.action_space.seed(args.seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    low, high = env.action_space.low, env.action_space.high
    actor = Actor(obs_dim, act_dim, args.sac_hidden, low, high).to(device)
    critic = Critic(obs_dim, act_dim, args.sac_hidden).to(device)
    target_critic = Critic(obs_dim, act_dim, args.sac_hidden).to(device)
    target_critic.load_state_dict(critic.state_dict())
    for p in target_critic.parameters(): p.requires_grad_(False)
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    target_entropy = -float(act_dim)
    oa = torch.optim.Adam(actor.parameters(), lr=args.sac_lr)
    oc = torch.optim.Adam(critic.parameters(), lr=args.sac_lr)
    oalpha = torch.optim.Adam([log_alpha], lr=args.sac_lr)
    model = GaussianMLPEnsemble(obs_dim, act_dim, args.model_hidden, args.model_layers, args.ensemble_size).to(device)
    om = torch.optim.Adam(model.parameters(), lr=args.model_lr, weight_decay=args.model_weight_decay)
    env_buf = ReplayBuffer(args.env_buffer_size, obs_dim, act_dim, device)
    model_buf = ReplayBuffer(args.model_buffer_size, obs_dim, act_dim, device)
    if args.track:
        import wandb; wandb.init(project=args.wandb_project, name=args.exp_name, config=vars(args))
    obs, _ = env.reset(seed=args.seed)
    elite_inds = list(range(args.elite_size))
    best, start = -float("inf"), time.time()
    for step in range(1, args.total_steps + 1):
        if step % 500 == 0:
            print(f"  step {step}/{args.total_steps}  env_buf={env_buf.size}  "
                  f"model_buf={model_buf.size}", flush=True)
        if step < args.init_random_steps:
            a = env.action_space.sample()
        else:
            with torch.no_grad():
                o = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                ax, _ = actor.sample(o); a = ax.cpu().numpy()[0]
        no, r, term, trunc, _ = env.step(a)
        env_buf.add(obs, a, r, no, term)
        obs = no if not (term or trunc) else env.reset()[0]
        if step >= args.init_random_steps and step % args.model_train_freq == 0:
            stats = train_dynamics(model, om, env_buf, args, device)
            elite_inds = stats["elite_inds"]
        if step >= args.init_random_steps and step % args.rollout_freq == 0:
            rollout_model(actor, model, elite_inds, env_buf, model_buf,
                          current_rollout_length(step, args), args.rollout_batch_size, args, device)
        if step >= args.init_random_steps:
            for _ in range(args.sac_updates_per_step):
                sac_update(actor, critic, target_critic, log_alpha, target_entropy,
                           oa, oc, oalpha, env_buf, model_buf, args)
        if step % args.eval_every == 0:
            ret = evaluate(eval_env, actor, device, args.eval_episodes)
            best = max(best, ret)
            print(f"step={step:7d}  eval={ret:8.1f}  best={best:8.1f}  "
                  f"k={current_rollout_length(step, args)}  env={env_buf.size}  "
                  f"model={model_buf.size}  elapsed={(time.time()-start)/60:5.1f}m")
            if args.track: wandb.log({"eval/return": ret, "eval/best": best}, step=step)
    print(f" best={best:.2f}")
    torch.save(actor.state_dict(), f"{args.exp_name}_actor.pth")

if __name__ == "__main__":
    main(tyro.cli(Args))