from __future__ import annotation
import random
import time
from dataclasses import dataclass
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro

@dataclass
class Args:
    exp_name: str = "pets_cartpole"
    env_id: str = "CartPole-v1"
    seed: int = 1
    total_steps: int = 30_000
    init_random_steps: int = 2_000
    eval_every: int = 1_000
    eval_episodes: int = 5
    track: bool = False
    wandb_project: str = "cleanmbrl"
    ensemble_size: int = 5
    hidden: int = 200
    layers: int = 3
    lr: float = 1e-3
    weight_decay: float = 1e-5
    model_train_freq: int = 250
    model_max_epochs: int = 40
    model_max_epochs_since_update: int = 5
    val_split: float = 0.2
    model_batch_size: int = 256
    horizon: int = 25
    population: int = 1000
    uncertainty_kappa: float = 1.0
    discount: float = 0.99

def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

class EnsembleLinear(nn.Module):
    def __init__(self, in_dim, out_dim, E):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(E, in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(E, 1, out_dim))
        for i in range(E):
            nn.init.trunc_normal_(self.weight[i], std=1.0 / (2.0 * np.sqrt(in_dim)))
    def forward(self, x):
        return torch.bmm(x, self.weight) + self.bias

class DynamicsEnsemble(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden, layers, E):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.E = E
        in_dim = obs_dim + n_actions
        out_dim = obs_dim + 1 + 1  
        self.layers = nn.ModuleList()
        for i in range(layers):
            self.layers.append(EnsembleLinear(in_dim if i == 0 else hidden, hidden, E))
        self.head = EnsembleLinear(hidden, out_dim, E)
        self.register_buffer("in_mean", torch.zeros(in_dim))
        self.register_buffer("in_std", torch.ones(in_dim))

    def forward(self, obs, a_onehot):
        if obs.ndim == 2:
            obs = obs.unsqueeze(0).expand(self.E, -1, -1)
            a_onehot = a_onehot.unsqueeze(0).expand(self.E, -1, -1)
        x = torch.cat([obs, a_onehot], dim=-1)
        x = (x - self.in_mean) / (self.in_std + 1e-6)
        for layer in self.layers:
            x = F.silu(layer(x))
        out = self.head(x)
        delta = out[..., : self.obs_dim]
        reward = out[..., self.obs_dim : self.obs_dim + 1]
        term_logit = out[..., -1:]
        return delta, reward, term_logit
    def update_norm(self, obs_np, act_onehot_np):
        x = np.concatenate([obs_np, act_onehot_np], axis=-1)
        self.in_mean.copy_(torch.as_tensor(x.mean(0), dtype=torch.float32))
        self.in_std.copy_(torch.as_tensor(x.std(0) + 1e-6, dtype=torch.float32))

class Replay:
    def __init__(self, cap, obs_dim, n_actions):
        self.cap = cap; self.size = 0; self.idx = 0
        self.obs = np.zeros((cap, obs_dim), dtype=np.float32)
        self.act = np.zeros((cap,), dtype=np.int64)
        self.rew = np.zeros((cap, 1), dtype=np.float32)
        self.next_obs = np.zeros((cap, obs_dim), dtype=np.float32)
        self.done = np.zeros((cap, 1), dtype=np.float32)
        self.n_actions = n_actions
    def add(self, o, a, r, no, d):
        i = self.idx
        self.obs[i] = o; self.act[i] = a; self.rew[i] = r
        self.next_obs[i] = no; self.done[i] = float(d)
        self.idx = (self.idx + 1) % self.cap
        self.size = min(self.size + 1, self.cap)
    def all(self):
        N = self.size
        a_onehot = np.eye(self.n_actions, dtype=np.float32)[self.act[:N]]
        return self.obs[:N], a_onehot, self.rew[:N], self.next_obs[:N], self.done[:N]

def train_model(model: DynamicsEnsemble, optim, buf: Replay, args: Args, device):
    obs, a_onehot, rew, nobs, done = buf.all()
    N = len(obs)
    delta = nobs - obs
    target = np.concatenate([delta, rew, done], axis=-1)  
    model.update_norm(obs, a_onehot)
    perm = np.random.permutation(N)
    n_val = int(N * args.val_split)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    obs_t = torch.as_tensor(obs, device=device)
    a_t = torch.as_tensor(a_onehot, device=device)
    tgt_t = torch.as_tensor(target, device=device)
    val = (obs_t[val_idx], a_t[val_idx], tgt_t[val_idx])
    tr_obs, tr_a, tr_tgt = obs_t[tr_idx], a_t[tr_idx], tgt_t[tr_idx]
    n = len(tr_idx)
    boot = np.stack([np.random.randint(0, n, n) for _ in range(model.E)]) 
    best = [float("inf")] * model.E
    stale = 0
    for epoch in range(args.model_max_epochs):
        for i in range(model.E):
            np.random.shuffle(boot[i])
        for start in range(0, n, args.model_batch_size):
            end = min(start + args.model_batch_size, n)
            idx = torch.as_tensor(boot[:, start:end], device=device, dtype=torch.long)
            o = tr_obs[idx]; a = tr_a[idx]; y = tr_tgt[idx]
            delta_pred, rew_pred, term_logit = model(o, a)
            delta_tgt = y[..., : model.obs_dim]
            rew_tgt = y[..., model.obs_dim : model.obs_dim + 1]
            term_tgt = y[..., -1:]
            loss = (
                F.mse_loss(delta_pred, delta_tgt)
                + F.mse_loss(rew_pred, rew_tgt)
                + F.binary_cross_entropy_with_logits(term_logit, term_tgt)
            )
            optim.zero_grad(); loss.backward(); optim.step()
        with torch.no_grad():
            d, r, t = model(val[0], val[1])
            val_tgt = val[2].unsqueeze(0).expand_as(torch.cat([d, r, t], -1))
            val_mse = (torch.cat([d, r, t.sigmoid()], -1) - val_tgt).pow(2).mean(dim=(1, 2))
        improved = False
        for i in range(model.E):
            v = val_mse[i].item()
            if v < best[i] * 0.99:
                best[i] = v; improved = True
        stale = 0 if improved else stale + 1
        if stale >= args.model_max_epochs_since_update:
            break
    return {"model/val_mse": float(val_mse.mean()), "model/epochs": epoch + 1}

@torch.no_grad()
def plan_action(model: DynamicsEnsemble, obs_np: np.ndarray, args: Args, device) -> int:
    P, H, A = args.population, args.horizon, model.n_actions
    actions = np.random.randint(0, A, size=(P, H))
    obs_t = torch.as_tensor(obs_np, device=device, dtype=torch.float32)
    state = obs_t.unsqueeze(0).expand(P, -1).clone()
    alive = torch.ones(P, 1, device=device)
    returns = torch.zeros(P, device=device)
    gamma = args.discount
    for t in range(H):
        a = torch.as_tensor(actions[:, t], device=device, dtype=torch.long)
        a_onehot = F.one_hot(a, A).float()
        delta, reward, term_logit = model(state, a_onehot)          
        delta_mean = delta.mean(0)
        delta_std = delta.std(0).mean(-1, keepdim=True)              
        reward_mean = reward.mean(0)
        term_p = torch.sigmoid(term_logit).mean(0)
        step_value = reward_mean - args.uncertainty_kappa * delta_std
        returns = returns + (gamma ** t) * alive.squeeze(-1) * step_value.squeeze(-1)
        alive = alive * (1.0 - term_p)
        state = state + delta_mean
    best = int(torch.argmax(returns).item())
    return int(actions[best, 0])

def evaluate(env, model, args, device, n_eps):
    rets = []
    for _ in range(n_eps):
        obs, _ = env.reset()
        ret, done = 0.0, False
        while not done:
            a = plan_action(model, obs, args, device)
            obs, r, term, trunc, _ = env.step(a)
            ret += r; done = term or trunc
        rets.append(ret)
    return float(np.mean(rets))

def main(args: Args):
    set_seed(args.seed)
    device = get_device()
    print(f"[pets-cartpole] device={device}")
    env = gym.make(args.env_id)
    eval_env = gym.make(args.env_id)
    env.action_space.seed(args.seed)
    obs_dim = env.observation_space.shape[0]
    n_actions = int(env.action_space.n)
    model = DynamicsEnsemble(obs_dim, n_actions, args.hidden, args.layers, args.ensemble_size).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    buf = Replay(100_000, obs_dim, n_actions)
    if args.track:
        import wandb
        wandb.init(project=args.wandb_project, name=args.exp_name, config=vars(args))
    obs, _ = env.reset(seed=args.seed)
    best_eval = -float("inf")
    start = time.time()
    for step in range(1, args.total_steps + 1):
        if step < args.init_random_steps or buf.size < args.model_batch_size:
            a = env.action_space.sample()
        else:
            a = plan_action(model, obs, args, device)
        no, r, term, trunc, _ = env.step(a)
        buf.add(obs, a, r, no, term)
        obs = no if not (term or trunc) else env.reset()[0]
        if step >= args.init_random_steps and step % args.model_train_freq == 0:
            stats = train_model(model, optim, buf, args, device)
            if args.track:
                wandb.log(stats, step=step)
        if step % args.eval_every == 0 and step >= args.init_random_steps:
            ret = evaluate(eval_env, model, args, device, args.eval_episodes)
            best_eval = max(best_eval, ret)
            print(f"step={step:6d}  eval={ret:6.1f}  best={best_eval:6.1f}  buf={buf.size}  "
                  f"elapsed={(time.time()-start)/60:5.1f}m")
            if args.track:
                wandb.log({"eval/return": ret, "eval/best": best_eval}, step=step)
            if best_eval >= 195.0:
                print("[pets-cartpole] solved."); break
    print(f"best={best_eval:.2f}")

if __name__ == "__main__":
    main(tyro.cli(Args))