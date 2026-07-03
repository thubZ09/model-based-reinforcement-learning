from __future__ import annotations
from dataclasses import dataclass, field
from typing import NamedTuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .networks import (
    ConvDecoder,
    ConvEncoder,
    OneHotST,
    SymlogMSE,
    TwoHotSymlog,
    mlp,
    symlog,
)

class State(NamedTuple):
    h: torch.Tensor       
    z: torch.Tensor        
    logits: torch.Tensor   

def flat_stoch(z: torch.Tensor) -> torch.Tensor:
    return z.flatten(start_dim=-2)

def state_feat(s: State) -> torch.Tensor:
    return torch.cat([s.h, flat_stoch(s.z)], dim=-1)

@dataclass
class RSSMConfig:
    stoch_groups: int = 32
    stoch_classes: int = 32
    deter_dim: int = 512
    hidden: int = 512
    mlp_layers: int = 1
    uniform_mix: float = 0.01

class RSSM(nn.Module):
    def __init__(self, action_dim: int, embed_dim: int, cfg: RSSMConfig):
        super().__init__()
        self.cfg = cfg
        stoch_flat = cfg.stoch_groups * cfg.stoch_classes
        self.in_proj = nn.Sequential(
            nn.Linear(stoch_flat + action_dim, cfg.hidden),
            nn.LayerNorm(cfg.hidden),
            nn.SiLU(),
        )
        self.gru = nn.GRUCell(cfg.hidden, cfg.deter_dim)
        self.prior_head = mlp(cfg.deter_dim, stoch_flat, cfg.hidden, cfg.mlp_layers)
        self.posterior_head = mlp(cfg.deter_dim + embed_dim, stoch_flat, cfg.hidden, cfg.mlp_layers)
    def initial(self, batch_size: int, device: torch.device) -> State:
        h = torch.zeros(batch_size, self.cfg.deter_dim, device=device)
        z = torch.zeros(batch_size, self.cfg.stoch_groups, self.cfg.stoch_classes, device=device)
        return State(h=h, z=z, logits=torch.zeros_like(z))
    def _det_step(self, prev: State, action: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(torch.cat([flat_stoch(prev.z), action], dim=-1))
        return self.gru(x, prev.h)
    def _prior(self, h: torch.Tensor):
        logits = self.prior_head(h).view(*h.shape[:-1], self.cfg.stoch_groups, self.cfg.stoch_classes)
        z = OneHotST(logits, self.cfg.uniform_mix).sample()
        return z, logits
    def _posterior(self, h: torch.Tensor, embed: torch.Tensor):
        logits = self.posterior_head(torch.cat([h, embed], dim=-1)).view(
            *h.shape[:-1], self.cfg.stoch_groups, self.cfg.stoch_classes
        )
        z = OneHotST(logits, self.cfg.uniform_mix).sample()
        return z, logits
    def obs_step(self, prev: State, action: torch.Tensor, embed: torch.Tensor):
        h = self._det_step(prev, action)
        prior_z, prior_logits = self._prior(h)
        post_z, post_logits = self._posterior(h, embed)
        return State(h=h, z=prior_z, logits=prior_logits), State(h=h, z=post_z, logits=post_logits)
    def img_step(self, prev: State, action: torch.Tensor) -> State:
        h = self._det_step(prev, action)
        z, logits = self._prior(h)
        return State(h=h, z=z, logits=logits)
    def observe(
        self,
        embeds: torch.Tensor,   
        actions: torch.Tensor,   
        is_first: torch.Tensor,
        initial: State | None = None,
    ):
        B, T = embeds.shape[:2]
        device = embeds.device
        prev = initial if initial is not None else self.initial(B, device)
        priors_h, priors_z, priors_logits = [], [], []
        posts_h, posts_z, posts_logits = [], [], []
        for t in range(T):
            mask = (1.0 - is_first[:, t].float()).view(-1, 1)
            prev = State(
                h=prev.h * mask,
                z=prev.z * mask.unsqueeze(-1),
                logits=prev.logits * mask.unsqueeze(-1),
            )
            prior, post = self.obs_step(prev, actions[:, t], embeds[:, t])
            priors_h.append(prior.h); priors_z.append(prior.z); priors_logits.append(prior.logits)
            posts_h.append(post.h);   posts_z.append(post.z);   posts_logits.append(post.logits)
            prev = post
        prior_states = State(
            h=torch.stack(priors_h, 1),
            z=torch.stack(priors_z, 1),
            logits=torch.stack(priors_logits, 1),
        )
        post_states = State(
            h=torch.stack(posts_h, 1),
            z=torch.stack(posts_z, 1),
            logits=torch.stack(posts_logits, 1),
        )
        return prior_states, post_states

    def imagine(self, initial: State, actor_fn, horizon: int):
        state = initial
        states_h = [state.h]; states_z = [state.z]; states_logits = [state.logits]
        actions = []
        for _ in range(horizon):
            feat = state_feat(state)
            action = actor_fn(feat)
            actions.append(action)
            state = self.img_step(state, action)
            states_h.append(state.h)
            states_z.append(state.z)
            states_logits.append(state.logits)
        return (
            State(
                h=torch.stack(states_h, 0),
                z=torch.stack(states_z, 0),
                logits=torch.stack(states_logits, 0),
            ),
            torch.stack(actions, 0),
        )

@dataclass
class WorldModelConfig:
    cnn_depth: int = 32
    head_hidden: int = 512
    head_layers: int = 2
    num_bins: int = 255
    rssm: RSSMConfig = field(default_factory=RSSMConfig)

class WorldModel(nn.Module):
    def __init__(self, action_dim: int, cfg: WorldModelConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = ConvEncoder(cfg.cnn_depth)
        self.rssm = RSSM(action_dim, self.encoder.out_dim, cfg.rssm)
        feat_dim = cfg.rssm.deter_dim + cfg.rssm.stoch_groups * cfg.rssm.stoch_classes
        self.feat_dim = feat_dim
        self.decoder = ConvDecoder(feat_dim, cfg.cnn_depth)
        self.reward_head = mlp(feat_dim, cfg.num_bins, cfg.head_hidden, cfg.head_layers)
        self.continue_head = mlp(feat_dim, 1, cfg.head_hidden, cfg.head_layers)
    def loss(
        self,
        obs: torch.Tensor,       
        actions: torch.Tensor,  
        rewards: torch.Tensor,   
        continues: torch.Tensor, 
        is_first: torch.Tensor,  
        free_nats: float = 1.0,
        kl_dyn_weight: float = 0.5,
        kl_rep_weight: float = 0.1,
    ):
        embeds = self.encoder(obs)
        prior, post = self.rssm.observe(embeds, actions, is_first)
        feat = state_feat(post)
        recon_loss = -self.decoder(feat).log_prob(obs).mean()
        reward_loss = -TwoHotSymlog(self.reward_head(feat)).log_prob(rewards).mean()
        cont_loss = F.binary_cross_entropy_with_logits(
            self.continue_head(feat).squeeze(-1), continues
        )
        dyn = OneHotST(post.logits.detach()).kl(OneHotST(prior.logits))
        rep = OneHotST(post.logits).kl(OneHotST(prior.logits.detach()))
        dyn = dyn.clamp(min=free_nats).mean()
        rep = rep.clamp(min=free_nats).mean()
        kl_loss = kl_dyn_weight * dyn + kl_rep_weight * rep
        loss = recon_loss + reward_loss + cont_loss + kl_loss
        metrics = {
            "wm/recon": recon_loss.item(),
            "wm/reward": reward_loss.item(),
            "wm/cont": cont_loss.item(),
            "wm/kl_dyn": dyn.item(),
            "wm/kl_rep": rep.item(),
            "wm/total": loss.item(),
        }
        return loss, metrics, post