from __future__ import annotations
import os
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
    exp_name: str = "mbpo_halfcheetah"
    env_id: str = "HalfCheetah-v5"
    seed: int = 1
    total_steps: int = 400_000
    eval_every: int = 5_000
    eval_episodes: int = 5
    track: bool = False
    wandb_project: str = "cleanmbrl"
    wandb_entity: str | None = None
    sac_lr: float = 3e-4
    sac_gamma: float = 0.99
    sac_tau: float = 0.005
    sac_batch_size: int = 256
    sac_updates_per_step: int = 20      
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
    rollout_length_end: int = 5
    rollout_length_warmup: int = 20_000
    rollout_length_anneal: int = 100_000
    init_random_steps: int = 5_000
    env_buffer_size: int = 1_000_000
    model_buffer_size: int = 400_000

def get_device() -> torch.device:
    return torch.device("cpu")

def configure_device(device: torch.device) -> None:
    if device.type == "mps":
        torch.set_default_dtype(torch.float32)
    elif device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.mul_(1.0 - tau).add_(tau * s.data)

class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, act_dim: int, device: torch.device):
        self.capacity = int(capacity)
        self.device = device
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((self.capacity, act_dim), dtype=np.float32)
        self.rew = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros((self.capacity, 1), dtype=np.float32)
        self.idx = 0
        self.size = 0
    def add(self, obs, act, rew, next_obs, done):
        i = self.idx
        self.obs[i] = obs
        self.act[i] = act
        self.rew[i] = rew
        self.next_obs[i] = next_obs
        self.done[i] = float(done) if np.ndim(done) == 0 else float(np.squeeze(done))
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    def add_batch(self, obs, act, rew, next_obs, done):
        n = obs.shape[0]
        indices = np.arange(self.idx, self.idx + n) % self.capacity
        self.obs[indices] = obs
        self.act[indices] = act
        self.rew[indices] = rew.reshape(n, 1) if rew.ndim == 1 else rew
        self.next_obs[indices] = next_obs
        self.done[indices] = done.reshape(n, 1) if done.ndim == 1 else done
        self.idx = (self.idx + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        b = (
            torch.as_tensor(self.obs[idx], device=self.device),
            torch.as_tensor(self.act[idx], device=self.device),
            torch.as_tensor(self.rew[idx], device=self.device),
            torch.as_tensor(self.next_obs[idx], device=self.device),
            torch.as_tensor(self.done[idx], device=self.device),
        )
        return b
    def sample_states(self, batch_size: int) -> np.ndarray:
        idx = np.random.randint(0, self.size, size=batch_size)
        return self.obs[idx].copy()

class GaussianMLPEnsemble(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int, layers: int, ensemble_size: int):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.ensemble_size = ensemble_size
        self.output_dim = obs_dim + 1 
        in_dim = obs_dim + act_dim
        self.layers = nn.ModuleList()
        for i in range(layers):
            self.layers.append(EnsembleLinear(in_dim if i == 0 else hidden, hidden, ensemble_size))
        self.mean_head = EnsembleLinear(hidden, self.output_dim, ensemble_size)
        self.logvar_head = EnsembleLinear(hidden, self.output_dim, ensemble_size)
        self.max_logvar = nn.Parameter(torch.full((1, 1, self.output_dim), 0.5))
        self.min_logvar = nn.Parameter(torch.full((1, 1, self.output_dim), -10.0))
        self.register_buffer("input_mean", torch.zeros(in_dim))
        self.register_buffer("input_std", torch.ones(in_dim))

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.input_mean) / (self.input_std + 1e-6)
    def forward(self, obs: torch.Tensor, act: torch.Tensor):
        if obs.ndim == 2:
            obs = obs.unsqueeze(0).expand(self.ensemble_size, -1, -1)
            act = act.unsqueeze(0).expand(self.ensemble_size, -1, -1)
        x = torch.cat([obs, act], dim=-1)
        x = self._normalize(x)
        for layer in self.layers:
            x = F.silu(layer(x))
        mean = self.mean_head(x)
        logvar = self.logvar_head(x)
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        return mean, logvar
    def update_normalization(self, obs_np: np.ndarray, act_np: np.ndarray) -> None:
        x = np.concatenate([obs_np, act_np], axis=-1)
        mean = x.mean(axis=0)
        std = x.std(axis=0) + 1e-6
        self.input_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.input_std.copy_(torch.as_tensor(std, dtype=torch.float32))

class EnsembleLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, ensemble_size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(ensemble_size, in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(ensemble_size, 1, out_dim))
        for i in range(ensemble_size):
            nn.init.trunc_normal_(self.weight[i], std=1.0 / (2.0 * np.sqrt(in_dim)))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.bmm(x, self.weight) + self.bias

def gaussian_nll(mean: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    inv_var = torch.exp(-logvar)
    mse = (mean - target).pow(2) * inv_var
    return (mse + logvar).mean()

def train_dynamics(
    model: GaussianMLPEnsemble,
    optimizer: torch.optim.Optimizer,
    buf: ReplayBuffer,
    args: Args,
    device: torch.device,
) -> dict:
    N = buf.size
    obs = buf.obs[:N].copy()
    act = buf.act[:N].copy()
    rew = buf.rew[:N].copy()
    next_obs = buf.next_obs[:N].copy()
    delta = next_obs - obs
    target = np.concatenate([delta, rew], axis=-1)
    model.update_normalization(obs, act)
    perm = np.random.permutation(N)
    n_val = int(N * args.model_val_split)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    obs_t = torch.as_tensor(obs, device=device)
    act_t = torch.as_tensor(act, device=device)
    tgt_t = torch.as_tensor(target, device=device)
    val = (obs_t[val_idx], act_t[val_idx], tgt_t[val_idx])
    train_obs, train_act, train_tgt = obs_t[train_idx], act_t[train_idx], tgt_t[train_idx]
    n_train = len(train_idx)
    boot_idx = np.stack([
        np.random.randint(0, n_train, size=n_train) for _ in range(model.ensemble_size)
    ])  
    best_val = [float("inf")] * model.ensemble_size
    epochs_since_improve = 0
    bs = args.model_batch_size
    for epoch in range(args.model_max_epochs):
        for i in range(model.ensemble_size):
            np.random.shuffle(boot_idx[i])
        for start in range(0, n_train, bs):
            end = min(start + bs, n_train)
            idx = boot_idx[:, start:end]  
            idx_t = torch.as_tensor(idx, device=device, dtype=torch.long)
            o = train_obs[idx_t]           
            a = train_act[idx_t]
            y = train_tgt[idx_t]
            mean, logvar = model(o, a)
            loss = gaussian_nll(mean, logvar, y)
            loss = loss + 0.01 * (model.max_logvar.sum() - model.min_logvar.sum())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            mean, logvar = model(val[0], val[1])
            val_target = val[2].unsqueeze(0).expand_as(mean)
            val_mse = (mean - val_target).pow(2).mean(dim=(1, 2))  
        improved = False
        for i in range(model.ensemble_size):
            v = val_mse[i].item()
            if v < best_val[i] * 0.99:
                best_val[i] = v
                improved = True
        epochs_since_improve = 0 if improved else epochs_since_improve + 1
        if epochs_since_improve >= args.model_max_epochs_since_update:
            break
    val_mse_final = val_mse.cpu().numpy()
    elite_inds = np.argsort(val_mse_final)[: args.elite_size]
    return {
        "model/val_mse_mean": float(val_mse_final.mean()),
        "model/val_mse_elite": float(val_mse_final[elite_inds].mean()),
        "model/epochs": epoch + 1,
        "elite_inds": elite_inds.tolist(),
    }
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

    def sample(self, obs: torch.Tensor):
        mu, log_std = self(obs)
        std = log_std.exp()
        normal = Normal(mu, std)
        x = normal.rsample()
        y = torch.tanh(x)
        action = y * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x) - torch.log(self.action_scale * (1 - y.pow(2)) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob

class Critic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x), self.q2(x)

@torch.no_grad()
def rollout_model(
    actor: Actor,
    model: GaussianMLPEnsemble,
    elite_inds: list[int],
    env_buf: ReplayBuffer,
    model_buf: ReplayBuffer,
    rollout_length: int,
    n_start_states: int,
    device: torch.device,
) -> None:
    starts = env_buf.sample_states(n_start_states)
    obs = torch.as_tensor(starts, device=device)
    for _ in range(rollout_length):
        act, _ = actor.sample(obs)
        mean, logvar = model(obs, act)  
        std = (logvar * 0.5).exp()
        sample = mean + std * torch.randn_like(mean)
        choice = torch.as_tensor(
            np.random.choice(elite_inds, size=sample.shape[1]),
            device=device, dtype=torch.long,
        )
        pred = sample[choice, torch.arange(sample.shape[1], device=device)]
        delta = pred[:, :-1]
        reward = pred[:, -1:]
        next_obs = obs + delta
        done = torch.zeros_like(reward) 
        model_buf.add_batch(
            *[x.cpu().numpy() for x in (obs, act, reward, next_obs, done)]
        )
        obs = next_obs

def sac_update(
    actor: Actor,
    critic: Critic,
    target_critic: Critic,
    log_alpha: torch.Tensor,
    target_entropy: float,
    optim_actor, optim_critic, optim_alpha,
    env_buf: ReplayBuffer,
    model_buf: ReplayBuffer,
    args: Args,
) -> dict:
    bs = args.sac_batch_size
    n_real = int(bs * args.real_ratio)
    n_model = bs - n_real
    use_model = model_buf.size > 0 and n_model > 0
    if use_model:
        ro, ra, rr, rno, rd = env_buf.sample(n_real) if n_real > 0 else (None,) * 5
        mo, ma, mr, mno, md = model_buf.sample(n_model)
        if n_real > 0:
            obs = torch.cat([ro, mo]); act = torch.cat([ra, ma])
            rew = torch.cat([rr, mr]); next_obs = torch.cat([rno, mno])
            done = torch.cat([rd, md])
        else:
            obs, act, rew, next_obs, done = mo, ma, mr, mno, md
    else:
        obs, act, rew, next_obs, done = env_buf.sample(bs)
    alpha = log_alpha.exp().detach()
    with torch.no_grad():
        next_act, next_logp = actor.sample(next_obs)
        q1_t, q2_t = target_critic(next_obs, next_act)
        q_t = torch.min(q1_t, q2_t) - alpha * next_logp
        y = rew + args.sac_gamma * (1.0 - done) * q_t
    q1, q2 = critic(obs, act)
    critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
    optim_critic.zero_grad()
    critic_loss.backward()
    optim_critic.step()
    new_act, logp = actor.sample(obs)
    q1_new, q2_new = critic(obs, new_act)
    q_new = torch.min(q1_new, q2_new)
    actor_loss = (alpha * logp - q_new).mean()
    optim_actor.zero_grad()
    actor_loss.backward()
    optim_actor.step()
    alpha_loss = -(log_alpha * (logp.detach() + target_entropy)).mean()
    optim_alpha.zero_grad()
    alpha_loss.backward()
    optim_alpha.step()
    soft_update(target_critic, critic, args.sac_tau)
    return {
        "sac/critic_loss": critic_loss.item(),
        "sac/actor_loss": actor_loss.item(),
        "sac/alpha": alpha.item(),
        "sac/logp_mean": logp.mean().item(),
    }

@torch.no_grad()
def evaluate(env: gym.Env, actor: Actor, device: torch.device, n_episodes: int) -> float:
    returns = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_ret, done = 0.0, False
        while not done:
            o = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            mu, _ = actor(o)
            a = (torch.tanh(mu) * actor.action_scale + actor.action_bias).cpu().numpy()[0]
            obs, r, term, trunc, _ = env.step(a)
            ep_ret += r
            done = term or trunc
        returns.append(ep_ret)
    return float(np.mean(returns))

def current_rollout_length(step: int, args: Args) -> int:
    if step < args.rollout_length_warmup:
        return args.rollout_length_start
    if step >= args.rollout_length_warmup + args.rollout_length_anneal:
        return args.rollout_length_end
    frac = (step - args.rollout_length_warmup) / args.rollout_length_anneal
    return int(round(args.rollout_length_start + frac * (args.rollout_length_end - args.rollout_length_start)))

def main(args: Args) -> None:
    set_seed(args.seed)
    device = get_device()
    configure_device(device)
    print(f"[mbpo] device={device} env={args.env_id} seed={args.seed}") 
    env = gym.make(args.env_id)
    eval_env = gym.make(args.env_id)
    env.action_space.seed(args.seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    action_low = env.action_space.low
    action_high = env.action_space.high
    actor = Actor(obs_dim, act_dim, args.sac_hidden, action_low, action_high).to(device)
    critic = Critic(obs_dim, act_dim, args.sac_hidden).to(device)
    target_critic = Critic(obs_dim, act_dim, args.sac_hidden).to(device)
    target_critic.load_state_dict(critic.state_dict())
    for p in target_critic.parameters():
        p.requires_grad_(False)
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    target_entropy = -float(act_dim)
    optim_actor = torch.optim.Adam(actor.parameters(), lr=args.sac_lr)
    optim_critic = torch.optim.Adam(critic.parameters(), lr=args.sac_lr)
    optim_alpha = torch.optim.Adam([log_alpha], lr=args.sac_lr)
    model = GaussianMLPEnsemble(
        obs_dim, act_dim, args.model_hidden, args.model_layers, args.ensemble_size
    ).to(device)
    optim_model = torch.optim.Adam(
        model.parameters(), lr=args.model_lr, weight_decay=args.model_weight_decay
    )
    env_buf = ReplayBuffer(args.env_buffer_size, obs_dim, act_dim, device)
    model_buf = ReplayBuffer(args.model_buffer_size, obs_dim, act_dim, device)
    if args.track:
        import wandb
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.exp_name, config=vars(args))
    obs, _ = env.reset(seed=args.seed)
    elite_inds: list[int] = list(range(args.elite_size))
    start_time = time.time()
    best_eval = -float("inf")
    for step in range(1, args.total_steps + 1):
        if step % 500 == 0:
            print(f"  step {step}/{args.total_steps}  env_buf={env_buf.size}  "
                  f"model_buf={model_buf.size}", flush=True)
        if step < args.init_random_steps:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                o = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                a, _ = actor.sample(o)
                action = a.cpu().numpy()[0]
        next_obs, reward, term, trunc, _ = env.step(action)
        done = term  
        env_buf.add(obs, action, reward, next_obs, done)
        obs = next_obs if not (term or trunc) else env.reset()[0]
        if step >= args.init_random_steps and step % args.model_train_freq == 0:
            stats = train_dynamics(model, optim_model, env_buf, args, device)
            elite_inds = stats["elite_inds"]
            if args.track:
                wandb.log({k: v for k, v in stats.items() if k != "elite_inds"}, step=step)
        if step >= args.init_random_steps and step % args.rollout_freq == 0:
            k = current_rollout_length(step, args)
            rollout_model(
                actor, model, elite_inds, env_buf, model_buf,
                rollout_length=k, n_start_states=args.rollout_batch_size, device=device,
            )
        if step >= args.init_random_steps:
            for update_i in range(args.sac_updates_per_step):
                stats = sac_update(
                    actor, critic, target_critic, log_alpha, target_entropy,
                    optim_actor, optim_critic, optim_alpha,
                    env_buf, model_buf, args,
            )
            if args.track and step % 500 == 0:
                wandb.log(stats, step=step)
        if step % args.eval_every == 0:
            ret = evaluate(eval_env, actor, device, args.eval_episodes)
            best_eval = max(best_eval, ret)
            elapsed = (time.time() - start_time) / 60
            print(
                f"step={step:7d}  eval={ret:8.1f}  best={best_eval:8.1f}  "
                f"k={current_rollout_length(step, args)}  env_buf={env_buf.size}  "
                f"model_buf={model_buf.size}  elapsed={elapsed:5.1f}m"
            )
            if args.track:
                wandb.log({"eval/return": ret, "eval/best": best_eval}, step=step)
    print(f"[best eval return = {best_eval:.2f}")
    torch.save(actor.state_dict(), f"{args.exp_name}_actor.pth")

if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)