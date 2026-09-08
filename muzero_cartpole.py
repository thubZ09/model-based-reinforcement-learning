from __future__ import annotations
import math
import random
import time
from dataclasses import dataclass, field
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro

@dataclass
class Args:
    exp_name: str = "muzero_cartpole"
    env_id: str = "CartPole-v1"
    seed: int = 1
    total_steps: int = 40_000
    eval_every: int = 2_000
    eval_episodes: int = 5
    track: bool = False
    wandb_project: str = "cleanmbrl"
    hidden_dim: int = 64
    repr_layers: int = 2
    dynamics_layers: int = 2
    prediction_layers: int = 2
    support_size: int = 25
    num_simulations: int = 50
    discount: float = 0.997
    root_dirichlet_alpha: float = 0.25
    root_exploration_fraction: float = 0.25
    pb_c_base: float = 19_652.0
    pb_c_init: float = 1.25
    temperature_init: float = 1.0
    temperature_final: float = 0.25
    temperature_decay_steps: int = 20_000
    lr: float = 2e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    unroll_steps: int = 5
    n_step: int = 10
    train_every: int = 1
    warmup_steps: int = 1_000
    grad_clip: float = 5.0
    value_loss_weight: float = 0.25
    cont_loss_weight: float = 1.0
    buffer_size: int = 50_000
    max_buffer_transitions: int = 200_000

def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

def mlp(in_dim, out_dim, hidden, layers):
    mods = [nn.Linear(in_dim, hidden), nn.ReLU()]
    for _ in range(layers - 1):
        mods += [nn.Linear(hidden, hidden), nn.ReLU()]
    mods.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*mods)

def scalar_to_support(x: torch.Tensor, support_size: int) -> torch.Tensor:
    eps = 0.001
    x = torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + eps * x
    x = x.clamp(-support_size, support_size)
    floor = x.floor()
    prob_upper = x - floor
    prob_lower = 1.0 - prob_upper
    floor_idx = floor.long() + support_size
    upper_idx = (floor_idx + 1).clamp(max=2 * support_size)
    out = torch.zeros(*x.shape, 2 * support_size + 1, device=x.device)
    out.scatter_(-1, floor_idx.unsqueeze(-1), prob_lower.unsqueeze(-1))
    out.scatter_(-1, upper_idx.unsqueeze(-1), prob_upper.unsqueeze(-1))
    return out

def support_to_scalar(logits: torch.Tensor, support_size: int) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    support = torch.arange(-support_size, support_size + 1,
                           device=logits.device, dtype=probs.dtype)
    x = (probs * support).sum(-1)
    eps = 0.001
    return torch.sign(x) * (
        ((torch.sqrt(1 + 4 * eps * (torch.abs(x) + 1 + eps)) - 1) / (2 * eps)) ** 2 - 1
    )

class MuZeroNet(nn.Module):
    def __init__(self, obs_dim: int, num_actions: int, args: Args):
        super().__init__()
        self.num_actions = num_actions
        self.support_size = args.support_size
        full = 2 * args.support_size + 1
        H = args.hidden_dim
        self.representation = mlp(obs_dim, H, H, args.repr_layers)
        self.dynamics_state = mlp(H + num_actions, H, H, args.dynamics_layers)
        self.dynamics_reward = mlp(H + num_actions, full, H, args.dynamics_layers)
        self.dynamics_cont = mlp(H + num_actions, 1, H, args.dynamics_layers)
        self.policy_head = mlp(H, num_actions, H, args.prediction_layers)
        self.value_head = mlp(H, full, H, args.prediction_layers)

    @staticmethod
    def _normalize(s: torch.Tensor) -> torch.Tensor:
        s_min = s.min(dim=-1, keepdim=True).values
        s_max = s.max(dim=-1, keepdim=True).values
        return (s - s_min) / (s_max - s_min).clamp(min=1e-5)
    def initial(self, obs: torch.Tensor):
        s = self._normalize(self.representation(obs))
        return s, self.policy_head(s), self.value_head(s)

    def recurrent(self, s: torch.Tensor, action: torch.Tensor):
        a = F.one_hot(action, self.num_actions).float()
        sa = torch.cat([s, a], dim=-1)
        s_next = self._normalize(self.dynamics_state(sa))
        return (s_next, self.dynamics_reward(sa), self.dynamics_cont(sa),
                self.policy_head(s_next), self.value_head(s_next))

class Node:
    __slots__ = ("prior", "value_sum", "visit_count", "children",
                 "reward", "cont", "hidden_state")
    def __init__(self, prior: float):
        self.prior = prior
        self.value_sum = 0.0
        self.visit_count = 0
        self.children: dict[int, Node] = {}
        self.reward = 0.0
        self.cont = 1.0
        self.hidden_state = None
    def expanded(self) -> bool:
        return len(self.children) > 0

    def value(self) -> float:
        return 0.0 if self.visit_count == 0 else self.value_sum / self.visit_count

class MinMaxStats:
    def __init__(self):
        self.maximum = -float("inf")
        self.minimum = float("inf")
    def update(self, v: float):
        self.maximum = max(self.maximum, v)
        self.minimum = min(self.minimum, v)
    def normalize(self, v: float) -> float:
        if self.maximum > self.minimum:
            return (v - self.minimum) / (self.maximum - self.minimum)
        return v

def ucb_score(parent: Node, child: Node, mm: MinMaxStats, args: Args) -> float:
    pb_c = math.log((parent.visit_count + args.pb_c_base + 1) / args.pb_c_base) + args.pb_c_init
    pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)
    prior_score = pb_c * child.prior
    if child.visit_count > 0:
        q = child.reward + args.discount * child.cont * child.value()
        value_score = mm.normalize(q)
    else:
        value_score = mm.normalize(parent.value())
    return prior_score + value_score


def select_child(node: Node, mm: MinMaxStats, args: Args):
    return max(node.children.items(), key=lambda kv: ucb_score(node, kv[1], mm, args))

def expand_node(node: Node, hidden, reward: float, cont: float,
                policy_logits: torch.Tensor, num_actions: int):
    node.hidden_state = hidden
    node.reward = reward
    node.cont = cont
    probs = F.softmax(policy_logits, dim=-1).cpu().numpy()
    for a in range(num_actions):
        node.children[a] = Node(prior=float(probs[a]))

def backpropagate(path: list[Node], value: float, mm: MinMaxStats, args: Args):
    for node in reversed(path):
        node.value_sum += value
        node.visit_count += 1
        mm.update(node.reward + args.discount * node.cont * node.value())
        value = node.reward + args.discount * node.cont * value

def add_root_noise(root: Node, args: Args):
    actions = list(root.children.keys())
    noise = np.random.dirichlet([args.root_dirichlet_alpha] * len(actions))
    for a, n in zip(actions, noise):
        c = root.children[a]
        c.prior = c.prior * (1 - args.root_exploration_fraction) + n * args.root_exploration_fraction

@torch.no_grad()
def run_mcts(net: MuZeroNet, obs: np.ndarray, args: Args, device, add_noise: bool) -> Node:
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    hidden, policy_logits, value_logits = net.initial(obs_t)
    root = Node(prior=1.0)
    expand_node(root, hidden, 0.0, 1.0, policy_logits[0], net.num_actions)
    root.value_sum = support_to_scalar(value_logits, args.support_size).item()
    root.visit_count = 1
    if add_noise:
        add_root_noise(root, args)
    mm = MinMaxStats()
    mm.update(root.value())
    for _ in range(args.num_simulations):
        node = root
        path = [node]
        actions: list[int] = []
        while node.expanded():
            a, node = select_child(node, mm, args)
            actions.append(a)
            path.append(node)
            if len(actions) > 50:
                break
        if not actions:
            continue
        parent = path[-2]
        a_t = torch.tensor([actions[-1]], device=device, dtype=torch.long)
        s_next, r_logits, c_logit, p_logits, v_logits = net.recurrent(parent.hidden_state, a_t)
        reward = support_to_scalar(r_logits, args.support_size).item()
        cont = torch.sigmoid(c_logit).item()
        value = support_to_scalar(v_logits, args.support_size).item()
        expand_node(node, s_next, reward, cont, p_logits[0], net.num_actions)
        backpropagate(path, value, mm, args)
    return root

@dataclass
class Trajectory:
    obs: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    policies: list = field(default_factory=list)
    values: list = field(default_factory=list)
    dones: list = field(default_factory=list)     
    def __len__(self):
        return len(self.actions)

class TrajectoryBuffer:
    def __init__(self, max_traj: int, max_transitions: int):
        self.trajectories: list[Trajectory] = []
        self.max_traj = max_traj
        self.max_transitions = max_transitions
        self.total = 0
    def add(self, traj: Trajectory):
        self.trajectories.append(traj)
        self.total += len(traj)
        while self.total > self.max_transitions or len(self.trajectories) > self.max_traj:
            self.total -= len(self.trajectories.pop(0))

    def sample(self, batch_size, unroll, n_step, num_actions, gamma):
        obs_b, act_b, rew_b, val_b, pol_b, mask_b, cont_b = [], [], [], [], [], [], []
        for _ in range(batch_size):
            traj = random.choice(self.trajectories)
            T = len(traj)
            start = np.random.randint(0, T)
            obs_b.append(traj.obs[start])
            actions, rewards, values, policies, masks, conts = [], [], [], [], [], []
            for k in range(unroll + 1):
                idx = start + k
                if idx < T:
                    v = 0.0
                    terminated = False
                    for j in range(idx, min(idx + n_step, T)):
                        v += traj.rewards[j] * (gamma ** (j - idx))
                        if traj.dones[j]:
                            terminated = True
                            break
                    if not terminated:
                        b = idx + n_step
                        if b < T:
                            v += traj.values[b] * (gamma ** n_step)
                    values.append(v)
                    policies.append(traj.policies[idx])
                    masks.append(1.0)
                else:
                    values.append(0.0)
                    policies.append(np.ones(num_actions) / num_actions)
                    masks.append(0.0)
                if k < unroll:
                    if idx < T:
                        actions.append(traj.actions[idx])
                        rewards.append(traj.rewards[idx])
                        conts.append(0.0 if traj.dones[idx] else 1.0)
                    else:
                        actions.append(np.random.randint(0, num_actions))
                        rewards.append(0.0)
                        conts.append(0.0)
            obs_b_last = None
            act_b.append(actions); rew_b.append(rewards); val_b.append(values)
            pol_b.append(policies); mask_b.append(masks); cont_b.append(conts)
        return {
            "obs": np.asarray(obs_b, dtype=np.float32),
            "actions": np.asarray(act_b, dtype=np.int64),
            "target_rewards": np.asarray(rew_b, dtype=np.float32),
            "target_values": np.asarray(val_b, dtype=np.float32),
            "target_policies": np.asarray(pol_b, dtype=np.float32),
            "target_conts": np.asarray(cont_b, dtype=np.float32),
            "mask": np.asarray(mask_b, dtype=np.float32),
        }

def train_step(net: MuZeroNet, opt, buf: TrajectoryBuffer, args: Args, device):
    b = buf.sample(args.batch_size, args.unroll_steps, args.n_step,
                   net.num_actions, args.discount)
    obs = torch.as_tensor(b["obs"], device=device)
    actions = torch.as_tensor(b["actions"], device=device)
    t_rew = torch.as_tensor(b["target_rewards"], device=device)
    t_val = torch.as_tensor(b["target_values"], device=device)
    t_pol = torch.as_tensor(b["target_policies"], device=device)
    t_cont = torch.as_tensor(b["target_conts"], device=device)
    mask = torch.as_tensor(b["mask"], device=device)
    s, p_logits, v_logits = net.initial(obs)
    lv, lr_, lp, lc = [], [], [], []
    lv.append(-(scalar_to_support(t_val[:, 0], args.support_size)
                * F.log_softmax(v_logits, -1)).sum(-1) * mask[:, 0])
    lp.append(-(t_pol[:, 0] * F.log_softmax(p_logits, -1)).sum(-1) * mask[:, 0])
    for k in range(args.unroll_steps):
        s, r_logits, c_logit, p_logits, v_logits = net.recurrent(s, actions[:, k])
        s = 0.5 * s + 0.5 * s.detach()                   
        lr_.append(-(scalar_to_support(t_rew[:, k], args.support_size)
                     * F.log_softmax(r_logits, -1)).sum(-1) * mask[:, k])
        lc.append(F.binary_cross_entropy_with_logits(
            c_logit.squeeze(-1), t_cont[:, k], reduction="none") * mask[:, k])
        lv.append(-(scalar_to_support(t_val[:, k + 1], args.support_size)
                    * F.log_softmax(v_logits, -1)).sum(-1) * mask[:, k + 1])
        lp.append(-(t_pol[:, k + 1] * F.log_softmax(p_logits, -1)).sum(-1) * mask[:, k + 1])
    value_loss = torch.stack(lv, 1).mean()
    reward_loss = torch.stack(lr_, 1).mean()
    policy_loss = torch.stack(lp, 1).mean()
    cont_loss = torch.stack(lc, 1).mean()
    loss = (args.value_loss_weight * value_loss + reward_loss
            + policy_loss + args.cont_loss_weight * cont_loss)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
    opt.step()
    return {"loss/total": loss.item(), "loss/value": value_loss.item(),
            "loss/reward": reward_loss.item(), "loss/policy": policy_loss.item(),
            "loss/cont": cont_loss.item()}

def temperature(step: int, args: Args) -> float:
    if step >= args.temperature_decay_steps:
        return args.temperature_final
    frac = step / args.temperature_decay_steps
    return args.temperature_init + frac * (args.temperature_final - args.temperature_init)


def select_action(root: Node, temp: float, num_actions: int) -> int:
    visits = np.array([root.children[a].visit_count for a in range(num_actions)],
                      dtype=np.float64)
    if visits.sum() == 0:
        return int(np.random.randint(num_actions))
    if temp < 1e-3:
        return int(np.argmax(visits))
    v = visits ** (1.0 / temp)
    return int(np.random.choice(num_actions, p=v / v.sum()))

@torch.no_grad()
def evaluate(env, net, args, device, n_episodes):
    rets = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        ret, done = 0.0, False
        while not done:
            root = run_mcts(net, obs, args, device, add_noise=False)
            visits = np.array([root.children[a].visit_count for a in range(net.num_actions)])
            obs, r, term, trunc, _ = env.step(int(np.argmax(visits)))
            ret += r
            done = term or trunc
        rets.append(ret)
    return float(np.mean(rets))

def main(args: Args):
    set_seed(args.seed)
    device = get_device()
    print(f"[muzero] device={device}", flush=True)
    env = gym.make(args.env_id)
    eval_env = gym.make(args.env_id)
    env.action_space.seed(args.seed)
    obs_dim = env.observation_space.shape[0]
    num_actions = int(env.action_space.n)
    net = MuZeroNet(obs_dim, num_actions, args).to(device)
    print(f"[muzero] params: {sum(p.numel() for p in net.parameters())/1e6:.3f}M", flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    buf = TrajectoryBuffer(args.buffer_size, args.max_buffer_transitions)
    if args.track:
        import wandb
        wandb.init(project=args.wandb_project, name=args.exp_name, config=vars(args))
    obs, _ = env.reset(seed=args.seed)
    traj = Trajectory()
    best = -float("inf")
    start = time.time()
    stats = None
    for step in range(1, args.total_steps + 1):
        root = run_mcts(net, obs, args, device, add_noise=True)
        visits = np.array([root.children[a].visit_count for a in range(num_actions)],
                          dtype=np.float64)
        policy = visits / max(visits.sum(), 1.0)
        action = select_action(root, temperature(step, args), num_actions)
        traj.obs.append(obs.astype(np.float32))
        traj.actions.append(action)
        traj.policies.append(policy.astype(np.float32))
        traj.values.append(float(root.value()))
        next_obs, reward, term, trunc, _ = env.step(action)
        traj.rewards.append(float(reward))
        traj.dones.append(bool(term))
        if term or trunc:
            buf.add(traj)
            traj = Trajectory()
            obs, _ = env.reset()
        else:
            obs = next_obs
        if step >= args.warmup_steps and buf.trajectories and step % args.train_every == 0:
            stats = train_step(net, opt, buf, args, device)
            if args.track and step % 500 == 0:
                wandb.log(stats, step=step)

        if step % args.eval_every == 0:
            ret = evaluate(eval_env, net, args, device, args.eval_episodes)
            best = max(best, ret)
            msg = (f"step={step:6d}  eval={ret:6.1f}  best={best:6.1f}  "
                   f"trajs={len(buf.trajectories)}  t={(time.time()-start)/60:.1f}m")
            if stats:
                msg += (f"  v={stats['loss/value']:.3f} p={stats['loss/policy']:.3f}"
                        f" c={stats['loss/cont']:.3f}")
            print(msg, flush=True)
            if args.track:
                wandb.log({"eval/return": ret, "eval/best": best}, step=step)
            if best >= 195.0:
                print("[muzero] solved.", flush=True)
                break
    print(f"[muzero] done. best={best:.2f}", flush=True)

if __name__ == "__main__":
    main(tyro.cli(Args))