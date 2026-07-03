from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from .networks import TwoHotSymlog, mlp
from .world_model import State, state_feat

@dataclass
class ActorCriticConfig:
    hidden: int = 512
    layers: int = 2
    num_bins: int = 255
    horizon: int = 15
    gamma: float = 0.997
    lambda_: float = 0.95
    actor_entropy: float = 3e-4
    critic_ema_tau: float = 0.02
    return_norm_decay: float = 0.99
    return_norm_limit: float = 1.0

class TanhNormalActor(nn.Module):
    def __init__(self, feat_dim: int, act_dim: int, action_low, action_high):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(feat_dim, 512), nn.LayerNorm(512), nn.SiLU(),
            nn.Linear(512, 512), nn.LayerNorm(512), nn.SiLU(),
        )
        self.mu_head = nn.Linear(512, act_dim)
        self.log_std_head = nn.Linear(512, act_dim)
        self.register_buffer("scale", torch.as_tensor((action_high - action_low) / 2.0, dtype=torch.float32))
        self.register_buffer("bias",  torch.as_tensor((action_high + action_low) / 2.0, dtype=torch.float32))

    def dist(self, feat: torch.Tensor):
        h = self.trunk(feat)
        mu = self.mu_head(h)
        log_std = self.log_std_head(h).clamp(-5.0, 2.0)
        return Normal(mu, log_std.exp())
    def sample(self, feat: torch.Tensor):
        d = self.dist(feat)
        x = d.rsample()
        y = torch.tanh(x)
        action = y * self.scale + self.bias
        logp = (d.log_prob(x) - torch.log(self.scale * (1 - y.pow(2)) + 1e-6)).sum(-1)
        ent = d.entropy().sum(-1)
        return action, logp, ent
    def deterministic(self, feat: torch.Tensor):
        h = self.trunk(feat)
        return torch.tanh(self.mu_head(h)) * self.scale + self.bias

class Critic(nn.Module):
    def __init__(self, feat_dim: int, cfg: ActorCriticConfig):
        super().__init__()
        self.net = mlp(feat_dim, cfg.num_bins, cfg.hidden, cfg.layers)
        self.cfg = cfg

    def dist(self, feat: torch.Tensor) -> TwoHotSymlog:
        return TwoHotSymlog(self.net(feat))

@torch.no_grad()
def ema_update(target: nn.Module, source: nn.Module, tau: float):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.mul_(1 - tau).add_(tau * s.data)

def lambda_return(
    rewards: torch.Tensor,    
    values: torch.Tensor,    
    continues: torch.Tensor, 
    gamma: float,
    lambda_: float,
) -> torch.Tensor:
    H = rewards.shape[0]
    inputs = rewards + gamma * continues[1:] * values[1:] * (1 - lambda_)
    last = values[-1]
    outputs = []
    for t in reversed(range(H)):
        last = inputs[t] + gamma * continues[t + 1] * lambda_ * last
        outputs.append(last)
    return torch.stack(list(reversed(outputs)), dim=0)   

class ReturnNormalizer:
    def __init__(self, decay: float, limit: float):
        self.decay = decay
        self.limit = limit
        self.std = 1.0
        self.initialized = False
    def update(self, returns: torch.Tensor):
        with torch.no_grad():
            r5, r95 = torch.quantile(
                returns.flatten(),
                torch.tensor([0.05, 0.95], device=returns.device),
            )
            span = max((r95 - r5).item(), 1.0)
            if not self.initialized:
                self.std = span
                self.initialized = True
            else:
                self.std = self.decay * self.std + (1 - self.decay) * span
    def scale(self) -> float:
        return max(self.std, self.limit)

def compute_ac_loss(
    actor: TanhNormalActor,
    critic: Critic,
    target_critic: Critic,
    rssm,
    init_state: State,
    reward_head: nn.Module,
    continue_head: nn.Module,
    cfg: ActorCriticConfig,
    return_norm: ReturnNormalizer,
):
    H = cfg.horizon
    h0 = init_state.h.reshape(-1, init_state.h.shape[-1])
    z0 = init_state.z.reshape(-1, *init_state.z.shape[-2:])
    s0 = State(h=h0, z=z0, logits=init_state.logits.reshape(-1, *init_state.logits.shape[-2:]))
    def actor_fn(feat):
        a, _, _ = actor.sample(feat)
        return a
    states, _ = rssm.imagine(s0, actor_fn, H)         
    feats = state_feat(states)                          
    rewards = TwoHotSymlog(reward_head(feats)).mean()   
    continues = torch.sigmoid(continue_head(feats)).squeeze(-1)
    values = target_critic.dist(feats).mean()            
    lam_ret = lambda_return(rewards[:-1], values, continues, cfg.gamma, cfg.lambda_)  
    critic_loss = -critic.dist(feats[:-1].detach()).log_prob(lam_ret.detach()).mean()
    return_norm.update(lam_ret.detach())
    norm = return_norm.scale()
    _, _, ent = actor.sample(feats[:-1].detach())
    actor_loss = -(lam_ret / norm).mean() - cfg.actor_entropy * ent.mean()
    metrics = {
        "ac/critic_loss": critic_loss.item(),
        "ac/actor_loss": actor_loss.item(),
        "ac/return_mean": lam_ret.mean().item(),
        "ac/return_norm": norm,
        "ac/value_mean": values.mean().item(),
        "ac/entropy": ent.mean().item(),
    }
    return actor_loss, critic_loss, metrics