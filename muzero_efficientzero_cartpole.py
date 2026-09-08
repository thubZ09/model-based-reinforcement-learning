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
    exp_name: str = "muzero_efficientzero_cartpole"
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
    num_simulations: int = 25
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
    consistency_loss_weight: float = 2.0   
    consistency_proj_dim: int = 64
    buffer_size: int = 50_000
    max_buffer_transitions: int = 200_000

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

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
    support = torch.arange(-support_size, support_size + 1, device=logits.device, dtype=probs.dtype)
    x = (probs * support).sum(-1)
    eps = 0.001
    x = torch.sign(x) * (
        ((torch.sqrt(1 + 4 * eps * (torch.abs(x) + 1 + eps)) - 1) / (2 * eps)) ** 2 - 1
    )
    return x

class MuZeroNet(nn.Module):
    def __init__(self, obs_dim: int, num_actions: int, args: Args):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.support_size = args.support_size
        full_support = 2 * args.support_size + 1
        H = args.hidden_dim
        self.representation = mlp(obs_dim, H, H, args.repr_layers)
        self.dynamics_state = mlp(H + num_actions, H, H, args.dynamics_layers)
        self.dynamics_reward = mlp(H + num_actions, full_support, H, args.dynamics_layers)
        self.policy_head = mlp(H, num_actions, H, args.prediction_layers)
        self.value_head = mlp(H, full_support, H, args.prediction_layers)
        proj_dim = args.consistency_proj_dim
        self.projector = mlp(H, proj_dim, H, 1)
        self.predictor = mlp(proj_dim, proj_dim, proj_dim, 1)

    def initial(self, obs: torch.Tensor):
        s = self.representation(obs)
        s = self._normalize(s)
        policy_logits = self.policy_head(s)
        value_logits = self.value_head(s)
        return s, policy_logits, value_logits
    def recurrent(self, s: torch.Tensor, action: torch.Tensor):
        a_onehot = F.one_hot(action, self.num_actions).float()
        sa = torch.cat([s, a_onehot], dim=-1)
        s_next = self.dynamics_state(sa)
        s_next = self._normalize(s_next)
        reward_logits = self.dynamics_reward(sa)
        policy_logits = self.policy_head(s_next)
        value_logits = self.value_head(s_next)
        return s_next, reward_logits, policy_logits, value_logits
    def project(self, s: torch.Tensor, with_predictor: bool) -> torch.Tensor:
        p = self.projector(s)
        if with_predictor:
            p = self.predictor(p)
        return p
    @staticmethod
    def _normalize(s: torch.Tensor) -> torch.Tensor:
        s_min = s.min(dim=-1, keepdim=True).values
        s_max = s.max(dim=-1, keepdim=True).values
        scale = (s_max - s_min).clamp(min=1e-5)
        return (s - s_min) / scale

class Node:
    __slots__ = ("prior", "value_sum", "visit_count", "children", "reward", "hidden_state")
    def __init__(self, prior: float):
        self.prior = prior
        self.value_sum = 0.0
        self.visit_count = 0
        self.children: dict[int, Node] = {}
        self.reward = 0.0
        self.hidden_state: torch.Tensor | None = None
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
        value_score = mm.normalize(child.reward + args.discount * child.value())
    else:
        value_score = 0.0
    return prior_score + value_score

def select_child(node: Node, mm: MinMaxStats, args: Args):
    return max(node.children.items(), key=lambda kv: ucb_score(node, kv[1], mm, args))

def expand_node(node: Node, hidden, reward: float, policy_logits: torch.Tensor, num_actions: int):
    node.hidden_state = hidden
    node.reward = reward
    probs = F.softmax(policy_logits, dim=-1).cpu().numpy()
    for a in range(num_actions):
        node.children[a] = Node(prior=float(probs[a]))

def backpropagate(path: list[Node], value: float, mm: MinMaxStats, args: Args):
    for node in reversed(path):
        node.value_sum += value
        node.visit_count += 1
        mm.update(node.reward + args.discount * node.value())
        value = node.reward + args.discount * value

def add_root_noise(root: Node, args: Args):
    actions = list(root.children.keys())
    noise = np.random.dirichlet([args.root_dirichlet_alpha] * len(actions))
    for a, n in zip(actions, noise):
        c = root.children[a]
        c.prior = c.prior * (1 - args.root_exploration_fraction) + n * args.root_exploration_fraction

@torch.no_grad()
def run_mcts(net: MuZeroNet, obs: np.ndarray, args: Args, device, add_noise: bool):
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    hidden, policy_logits, _ = net.initial(obs_t)
    root = Node(prior=1.0)
    expand_node(root, hidden, 0.0, policy_logits[0], net.num_actions)
    if add_noise:
        add_root_noise(root, args)
    mm = MinMaxStats()
    for _ in range(args.num_simulations):
        node = root
        path = [node]
        action_history = []
        while node.expanded():
            action, node = select_child(node, mm, args)
            action_history.append(action)
            path.append(node)
            if len(action_history) > 50:
                break
        parent = path[-2]
        a = torch.tensor([action_history[-1]], device=device, dtype=torch.long)
        s_next, reward_logits, policy_logits, value_logits = net.recurrent(parent.hidden_state, a)
        reward = support_to_scalar(reward_logits, args.support_size).item()
        value = support_to_scalar(value_logits, args.support_size).item()
        expand_node(node, s_next, reward, policy_logits[0], net.num_actions)
        backpropagate(path, value, mm, args)
    return root

@dataclass
class Trajectory:
    obs: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    policies: list = field(default_factory=list)
    values: list = field(default_factory=list)
    def __len__(self):
        return len(self.actions)

class TrajectoryBuffer:
    def __init__(self, max_trajectories: int, max_total_transitions: int):
        self.trajectories: list[Trajectory] = []
        self.max_trajectories = max_trajectories
        self.max_total_transitions = max_total_transitions
        self.total = 0
    def add(self, traj: Trajectory):
        self.trajectories.append(traj)
        self.total += len(traj)
        while self.total > self.max_total_transitions or len(self.trajectories) > self.max_trajectories:
            old = self.trajectories.pop(0)
            self.total -= len(old)
    def sample(self, batch_size, unroll, n_step, num_actions, gamma):
        obs_b, actions_b, target_rewards_b = [], [], []
        target_values_b, target_policies_b, mask_b = [], [], []
        future_obs_b = []
        for _ in range(batch_size):
            traj = random.choice(self.trajectories)
            T = len(traj)
            start = np.random.randint(0, T)
            obs_b.append(traj.obs[start])
            actions, rewards, values, policies, masks, future_obs = [], [], [], [], [], []
            for k in range(unroll + 1):
                idx = start + k
                if idx < T:
                    bootstrap_idx = idx + n_step
                    if bootstrap_idx < T:
                        v = traj.values[bootstrap_idx] * (gamma ** n_step)
                    else:
                        v = 0.0
                    for j in range(idx, min(idx + n_step, T)):
                        v += traj.rewards[j] * (gamma ** (j - idx))
                    values.append(v)
                    policies.append(traj.policies[idx])
                    masks.append(1.0)
                    future_obs.append(traj.obs[idx])
                else:
                    values.append(0.0)
                    policies.append(np.ones(num_actions) / num_actions)
                    masks.append(0.0)
                    future_obs.append(traj.obs[-1])   
                if k < unroll:
                    if idx < T:
                        actions.append(traj.actions[idx])
                        rewards.append(traj.rewards[idx])
                    else:
                        actions.append(np.random.randint(0, num_actions))
                        rewards.append(0.0)
            actions_b.append(actions)
            target_rewards_b.append(rewards)
            target_values_b.append(values)
            target_policies_b.append(policies)
            mask_b.append(masks)
            future_obs_b.append(future_obs)
        return {
            "obs": np.asarray(obs_b, dtype=np.float32),
            "actions": np.asarray(actions_b, dtype=np.int64),
            "target_rewards": np.asarray(target_rewards_b, dtype=np.float32),
            "target_values": np.asarray(target_values_b, dtype=np.float32),
            "target_policies": np.asarray(target_policies_b, dtype=np.float32),
            "mask": np.asarray(mask_b, dtype=np.float32),
            "future_obs": np.asarray(future_obs_b, dtype=np.float32),
        }

def train_step(net: MuZeroNet, optimizer, buf: TrajectoryBuffer, args: Args, device):
    batch = buf.sample(args.batch_size, args.unroll_steps, args.n_step, net.num_actions, args.discount)
    obs = torch.as_tensor(batch["obs"], device=device)
    actions = torch.as_tensor(batch["actions"], device=device)
    target_rewards = torch.as_tensor(batch["target_rewards"], device=device)
    target_values = torch.as_tensor(batch["target_values"], device=device)
    target_policies = torch.as_tensor(batch["target_policies"], device=device)
    mask = torch.as_tensor(batch["mask"], device=device)
    future_obs = torch.as_tensor(batch["future_obs"], device=device)   
    s, policy_logits, value_logits = net.initial(obs)
    losses_value, losses_reward, losses_policy, losses_consistency = [], [], [], []
    v_target_support = scalar_to_support(target_values[:, 0], args.support_size)
    losses_value.append(-(v_target_support * F.log_softmax(value_logits, dim=-1)).sum(-1) * mask[:, 0])
    losses_policy.append(-(target_policies[:, 0] * F.log_softmax(policy_logits, dim=-1)).sum(-1) * mask[:, 0])
    for k in range(args.unroll_steps):
        s, reward_logits, policy_logits, value_logits = net.recurrent(s, actions[:, k])
        s = 0.5 * s + 0.5 * s.detach()
        with torch.no_grad():
            target_latent, _, _ = net.initial(future_obs[:, k + 1])
            target_proj = net.project(target_latent, with_predictor=False)
            target_proj = F.normalize(target_proj, dim=-1)
        online_proj = net.project(s, with_predictor=True)
        online_proj = F.normalize(online_proj, dim=-1)
        cos_loss = -(online_proj * target_proj).sum(-1) * mask[:, k + 1]
        losses_consistency.append(cos_loss)
        r_target_support = scalar_to_support(target_rewards[:, k], args.support_size)
        losses_reward.append(-(r_target_support * F.log_softmax(reward_logits, dim=-1)).sum(-1) * mask[:, k])
        v_target_support = scalar_to_support(target_values[:, k + 1], args.support_size)
        losses_value.append(-(v_target_support * F.log_softmax(value_logits, dim=-1)).sum(-1) * mask[:, k + 1])
        losses_policy.append(-(target_policies[:, k + 1] * F.log_softmax(policy_logits, dim=-1)).sum(-1) * mask[:, k + 1])
    value_loss = torch.stack(losses_value, dim=1).mean()
    reward_loss = torch.stack(losses_reward, dim=1).mean() if losses_reward else torch.tensor(0.0, device=device)
    policy_loss = torch.stack(losses_policy, dim=1).mean()
    consistency_loss = torch.stack(losses_consistency, dim=1).mean() if losses_consistency else torch.tensor(0.0, device=device)
    loss = (
        args.value_loss_weight * value_loss
        + reward_loss
        + policy_loss
        + args.consistency_loss_weight * consistency_loss
    )
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
    optimizer.step()
    return {
        "loss/total": loss.item(),
        "loss/value": value_loss.item(),
        "loss/reward": reward_loss.item(),
        "loss/policy": policy_loss.item(),
        "loss/consistency": consistency_loss.item(),
    }

@torch.no_grad()
def evaluate(env, net: MuZeroNet, args: Args, device, n_episodes: int):
    rets = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        ret, done = 0.0, False
        while not done:
            root = run_mcts(net, obs, args, device, add_noise=False)
            visits = np.array([root.children[a].visit_count for a in range(net.num_actions)])
            action = int(np.argmax(visits))
            obs, r, term, trunc, _ = env.step(action)
            ret += r
            done = term or trunc
        rets.append(ret)
    return float(np.mean(rets))

def temperature(step: int, args: Args) -> float:
    if step >= args.temperature_decay_steps:
        return args.temperature_final
    frac = step / args.temperature_decay_steps
    return args.temperature_init + frac * (args.temperature_final - args.temperature_init)

def select_action(root: Node, temp: float, num_actions: int) -> int:
    visits = np.array([root.children[a].visit_count for a in range(num_actions)], dtype=np.float64)
    if temp < 1e-3:
        return int(np.argmax(visits))
    visits = visits ** (1.0 / temp)
    probs = visits / visits.sum()
    return int(np.random.choice(num_actions, p=probs))

def main(args: Args):
    set_seed(args.seed)
    device = get_device()
    print(f"[muzero-ez] device={device}", flush=True)
    env = gym.make(args.env_id)
    eval_env = gym.make(args.env_id)
    env.action_space.seed(args.seed)
    obs_dim = env.observation_space.shape[0]
    num_actions = int(env.action_space.n)
    net = MuZeroNet(obs_dim, num_actions, args).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    buf = TrajectoryBuffer(args.buffer_size, args.max_buffer_transitions)
    if args.track:
        import wandb
        wandb.init(project=args.wandb_project, name=args.exp_name, config=vars(args))

    obs, _ = env.reset(seed=args.seed)
    traj = Trajectory()
    best_eval = -float("inf")
    start = time.time()
    for step in range(1, args.total_steps + 1):
        if step % 500 == 0:
            print(f"  step {step}/{args.total_steps}  trajs={len(buf.trajectories)}", flush=True)
        root = run_mcts(net, obs, args, device, add_noise=True)
        visits = np.array([root.children[a].visit_count for a in range(num_actions)], dtype=np.float64)
        policy = visits / visits.sum()
        action = select_action(root, temperature(step, args), num_actions)
        traj.obs.append(obs.astype(np.float32))
        traj.actions.append(action)
        traj.policies.append(policy.astype(np.float32))
        traj.values.append(float(root.value()))
        next_obs, reward, term, trunc, _ = env.step(action)
        traj.rewards.append(float(reward))
        if term or trunc:
            traj.obs.append(next_obs.astype(np.float32)) 
            buf.add(traj)
            traj = Trajectory()
            obs, _ = env.reset()
        else:
            obs = next_obs
        if step >= args.warmup_steps and len(buf.trajectories) > 0 and step % args.train_every == 0:
            stats = train_step(net, optimizer, buf, args, device)
            if args.track and step % 500 == 0:
                wandb.log(stats, step=step)
        if step % args.eval_every == 0:
            ret = evaluate(eval_env, net, args, device, args.eval_episodes)
            best_eval = max(best_eval, ret)
            elapsed = (time.time() - start) / 60
            print(f"step={step:6d}  eval={ret:6.1f}  best={best_eval:6.1f}  "
                  f"trajs={len(buf.trajectories)}  elapsed={elapsed:5.1f}m", flush=True)
            if args.track:
                wandb.log({"eval/return": ret, "eval/best": best_eval}, step=step)
            if best_eval >= 195.0:
                print("[muzero-ez] solved.", flush=True)
                break
    print(f"[muzero-ez] done. best={best_eval:.2f}", flush=True)

if __name__ == "__main__":
    main(tyro.cli(Args))