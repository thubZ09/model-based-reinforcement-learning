from __future__ import annotations
import random
import time
from dataclasses import dataclass
import numpy as np
import torch
import tyro
from .actor_critic import (
    ActorCriticConfig,
    Critic,
    ReturnNormalizer,
    TanhNormalActor,
    compute_ac_loss,
    ema_update,
)
from .env_wrappers import DMCPixelEnv
from .replay import Episode, EpisodicReplay
from .world_model import State, WorldModel, WorldModelConfig

@dataclass
class Args:
    exp_name: str = "dreamerv3_walker_walk"
    domain: str = "walker"
    task: str = "walk"
    seed: int = 1
    total_env_steps: int = 1_000_000
    action_repeat: int = 2
    image_size: int = 64
    track: bool = False
    wandb_project: str = "cleanmbrl"
    buffer_steps: int = 1_000_000
    seq_len: int = 64
    batch_size: int = 16
    train_every: int = 5
    warmup_env_steps: int = 5_000
    eval_every_env_steps: int = 10_000
    eval_episodes: int = 3
    wm_lr: float = 1e-4
    wm_grad_clip: float = 1000.0
    free_nats: float = 1.0
    kl_dyn_weight: float = 0.5
    kl_rep_weight: float = 0.1
    ac_lr: float = 3e-5
    ac_grad_clip: float = 100.0
    horizon: int = 15
    gamma: float = 0.997
    lambda_: float = 0.95
    actor_entropy: float = 3e-4
    critic_ema_tau: float = 0.02

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

@torch.no_grad()
def policy_step(
    world: WorldModel,
    actor: TanhNormalActor,
    obs_np: np.ndarray,
    state,
    prev_action: np.ndarray,
    is_first: bool,
    device,
):
    obs = (
        torch.as_tensor(np.ascontiguousarray(obs_np), dtype=torch.float32, device=device)
        .permute(2, 0, 1)
        .unsqueeze(0)
        / 255.0
        - 0.5
    )
    embed = world.encoder(obs)
    if state is None or is_first:
        state = world.rssm.initial(1, device)
    prev_a = torch.as_tensor(prev_action, dtype=torch.float32, device=device).unsqueeze(0)
    _, post = world.rssm.obs_step(state, prev_a, embed)
    from .world_model import state_feat
    feat = state_feat(post)
    action, _, _ = actor.sample(feat)
    return post, action.cpu().numpy()[0]

@torch.no_grad()
def evaluate(env, world, actor, device, n_episodes: int) -> float:
    from .world_model import state_feat
    returns = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        state = None
        prev_action = np.zeros(env.action_space.shape, dtype=np.float32)
        ret, done = 0.0, False
        is_first = True
        while not done:
            obs_t = (
                torch.as_tensor(np.ascontiguousarray(obs), dtype=torch.float32, device=device)
                .permute(2, 0, 1)
                .unsqueeze(0)
                / 255.0
                - 0.5
            )
            embed = world.encoder(obs_t)
            if state is None or is_first:
                state = world.rssm.initial(1, device)
            prev_a = torch.as_tensor(prev_action, dtype=torch.float32, device=device).unsqueeze(0)
            _, post = world.rssm.obs_step(state, prev_a, embed)
            action = actor.deterministic(state_feat(post)).cpu().numpy()[0]
            obs, r, term, trunc, _ = env.step(action)
            ret += r
            done = term or trunc
            state = post
            prev_action = action
            is_first = False
        returns.append(ret)
    return float(np.mean(returns))

def main(args: Args):
    set_seed(args.seed)
    device = get_device()
    print(f"[dreamerv3] device={device}  {args.domain}/{args.task}", flush=True)
    env = DMCPixelEnv(args.domain, args.task, args.image_size, args.action_repeat, seed=args.seed)
    eval_env = DMCPixelEnv(args.domain, args.task, args.image_size, args.action_repeat, seed=args.seed + 1000)
    act_dim = env.action_space.shape[0]
    wm_cfg = WorldModelConfig()
    world = WorldModel(act_dim, wm_cfg).to(device)
    feat_dim = world.feat_dim
    ac_cfg = ActorCriticConfig(
        horizon=args.horizon,
        gamma=args.gamma,
        lambda_=args.lambda_,
        actor_entropy=args.actor_entropy,
        critic_ema_tau=args.critic_ema_tau,
    )
    actor = TanhNormalActor(feat_dim, act_dim, env.action_space.low, env.action_space.high).to(device)
    critic = Critic(feat_dim, ac_cfg).to(device)
    target_critic = Critic(feat_dim, ac_cfg).to(device)
    target_critic.load_state_dict(critic.state_dict())
    for p in target_critic.parameters():
        p.requires_grad_(False)
    opt_wm = torch.optim.Adam(world.parameters(), lr=args.wm_lr)
    opt_actor = torch.optim.Adam(actor.parameters(), lr=args.ac_lr)
    opt_critic = torch.optim.Adam(critic.parameters(), lr=args.ac_lr)
    return_norm = ReturnNormalizer(decay=0.99, limit=1.0)
    buf = EpisodicReplay(args.buffer_steps, args.seq_len)
    if args.track:
        import wandb
        wandb.init(project=args.wandb_project, name=args.exp_name, config=vars(args))
    obs, _ = env.reset()
    ep = Episode()
    ep.obs.append(obs)
    ep.is_first.append(1.0)
    state = None
    prev_action = np.zeros(act_dim, dtype=np.float32)
    is_first = True
    env_steps = 0
    train_steps = 0
    best_eval = -float("inf")
    start = time.time()
    while env_steps < args.total_env_steps:
        if env_steps % 500 == 0:
            print(f"  env_steps={env_steps}  buf={buf.total}  train_steps={train_steps}", flush=True)
        if env_steps < args.warmup_env_steps:
            action = env.sample_action()
        else:
            state, action = policy_step(world, actor, ep.obs[-1], state, prev_action, is_first, device)
        next_obs, reward, term, trunc, _ = env.step(action)
        env_steps += 1
        ep.action.append(action.astype(np.float32))
        ep.reward.append(float(reward))
        ep.cont.append(0.0 if term else 1.0)
        if term or trunc:
            buf.add_episode(ep)
            obs, _ = env.reset()
            ep = Episode()
            ep.obs.append(obs)
            ep.is_first.append(1.0)
            state = None
            prev_action = np.zeros(act_dim, dtype=np.float32)
            is_first = True
        else:
            ep.obs.append(next_obs)
            ep.is_first.append(0.0)
            prev_action = action
            is_first = False
        if (
            env_steps >= args.warmup_env_steps
            and buf.ready()
            and env_steps % args.train_every == 0
        ):
            batch = buf.sample(args.batch_size, device)
            wm_loss, wm_metrics, post = world.loss(
                batch["obs"],
                batch["action"],
                batch["reward"],
                batch["cont"],
                batch["is_first"],
                free_nats=args.free_nats,
                kl_dyn_weight=args.kl_dyn_weight,
                kl_rep_weight=args.kl_rep_weight,
            )
            opt_wm.zero_grad()
            wm_loss.backward()
            torch.nn.utils.clip_grad_norm_(world.parameters(), args.wm_grad_clip)
            opt_wm.step()
            init = State(
                h=post.h.detach(),
                z=post.z.detach(),
                logits=post.logits.detach(),
            )
            actor_loss, critic_loss, ac_metrics = compute_ac_loss(
                actor, critic, target_critic,
                world.rssm, init,
                world.reward_head, world.continue_head,
                ac_cfg, return_norm,
            )
            opt_critic.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), args.ac_grad_clip)
            opt_critic.step()
            opt_actor.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), args.ac_grad_clip)
            opt_actor.step()
            ema_update(target_critic, critic, ac_cfg.critic_ema_tau)
            train_steps += 1
            if args.track and train_steps % 100 == 0:
                wandb.log({**wm_metrics, **ac_metrics}, step=env_steps)
        if env_steps % args.eval_every_env_steps == 0 and env_steps > 0:
            ret = evaluate(eval_env, world, actor, device, args.eval_episodes)
            best_eval = max(best_eval, ret)
            elapsed = (time.time() - start) / 60
            print(
                f"env_steps={env_steps:8d}  eval={ret:7.1f}  best={best_eval:7.1f}"
                f"  train_steps={train_steps:6d}  elapsed={elapsed:6.1f}m",
                flush=True,
            )
            if args.track:
                wandb.log({"eval/return": ret, "eval/best": best_eval}, step=env_steps)
    print(f"best={best_eval:.2f}", flush=True)

if __name__ == "__main__":
    main(tyro.cli(Args))