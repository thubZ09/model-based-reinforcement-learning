from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))
def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.expm1(torch.abs(x))

class TwoHotSymlog:
    def __init__(self, logits: torch.Tensor, vmin: float = -20.0, vmax: float = 20.0):
        self.logits = logits
        self.num_bins = logits.shape[-1]
        self.bins = torch.linspace(vmin, vmax, self.num_bins, device=logits.device, dtype=logits.dtype)
        self.vmin = vmin
        self.vmax = vmax
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        x_sym = symlog(x).clamp(self.vmin, self.vmax)
        idx = (x_sym - self.vmin) / (self.vmax - self.vmin) * (self.num_bins - 1)
        lo = idx.floor().long().clamp(0, self.num_bins - 1)
        hi = (lo + 1).clamp(max=self.num_bins - 1)
        w_hi = (idx - lo.float()).unsqueeze(-1)
        w_lo = 1.0 - w_hi
        target = torch.zeros_like(self.logits)
        target.scatter_(-1, lo.unsqueeze(-1), w_lo)
        target.scatter_(-1, hi.unsqueeze(-1), w_hi)
        return (target * F.log_softmax(self.logits, dim=-1)).sum(-1)
    def mean(self) -> torch.Tensor:
        probs = F.softmax(self.logits, dim=-1)
        return symexp((probs * self.bins).sum(-1))

class SymlogMSE:
    def __init__(self, pred: torch.Tensor):
        self.pred = pred
    def log_prob(self, target: torch.Tensor) -> torch.Tensor:
        diff = self.pred - symlog(target)
        return -0.5 * diff.pow(2).flatten(start_dim=-3).sum(-1)
    def mean(self) -> torch.Tensor:
        return symexp(self.pred)

class OneHotST:
    def __init__(self, logits: torch.Tensor, uniform_mix: float = 0.01):
        if uniform_mix > 0:
            probs = (1.0 - uniform_mix) * F.softmax(logits, dim=-1) + uniform_mix / logits.shape[-1]
            self.logits = torch.log(probs + 1e-8)
        else:
            self.logits = logits
    def sample(self) -> torch.Tensor:
        probs = F.softmax(self.logits, dim=-1)
        index = torch.distributions.Categorical(probs=probs).sample()
        hard = F.one_hot(index, num_classes=self.logits.shape[-1]).float()
        return hard + probs - probs.detach()
    def kl(self, other: "OneHotST") -> torch.Tensor:
        p = F.softmax(self.logits, dim=-1)
        log_p = F.log_softmax(self.logits, dim=-1)
        log_q = F.log_softmax(other.logits, dim=-1)
        return (p * (log_p - log_q)).sum(-1).sum(-1)

class ConvEncoder(nn.Module):
    def __init__(self, depth: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, depth, 4, 2),
            nn.LayerNorm([depth, 31, 31]),
            nn.SiLU(),
            nn.Conv2d(depth, 2 * depth, 4, 2),
            nn.LayerNorm([2 * depth, 14, 14]),
            nn.SiLU(),
            nn.Conv2d(2 * depth, 4 * depth, 4, 2),
            nn.LayerNorm([4 * depth, 6, 6]),
            nn.SiLU(),
            nn.Conv2d(4 * depth, 8 * depth, 4, 2),
            nn.LayerNorm([8 * depth, 2, 2]),
            nn.SiLU(),
        )
        self.out_dim = 8 * depth * 2 * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.view(-1, *shape[-3:])
        x = symlog(x)
        h = self.net(x).flatten(start_dim=1)
        return h.view(*shape[:-3], self.out_dim)

class ConvDecoder(nn.Module):
    def __init__(self, feat_dim: int, depth: int = 32):
        super().__init__()
        self.depth = depth
        self.fc = nn.Linear(feat_dim, 8 * depth * 2 * 2)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(8 * depth, 4 * depth, 4, 2),
            nn.LayerNorm([4 * depth, 6, 6]),
            nn.SiLU(),
            nn.ConvTranspose2d(4 * depth, 2 * depth, 4, 2),
            nn.LayerNorm([2 * depth, 14, 14]),
            nn.SiLU(),
            nn.ConvTranspose2d(2 * depth, depth, 4, 2),
            nn.LayerNorm([depth, 30, 30]),
            nn.SiLU(),
            nn.ConvTranspose2d(depth, 3, 6, 2),
        )
    def forward(self, feat: torch.Tensor) -> SymlogMSE:
        shape = feat.shape
        x = self.fc(feat).view(-1, 8 * self.depth, 2, 2)
        x = self.net(x).view(*shape[:-1], 3, 64, 64)
        return SymlogMSE(x)

def mlp(in_dim: int, out_dim: int, hidden: int, layers: int) -> nn.Sequential:
    mods: list[nn.Module] = []
    d = in_dim
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.SiLU()]
        d = hidden
    mods.append(nn.Linear(d, out_dim))
    return nn.Sequential(*mods)