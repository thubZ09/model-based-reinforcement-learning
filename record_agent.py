from __future__ import annotations
import argparse
from pathlib import Path
import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont

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
        self.register_buffer(
            "action_scale", torch.as_tensor((action_high - action_low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.as_tensor((action_high + action_low) / 2.0, dtype=torch.float32)
        )
    def forward(self, obs: torch.Tensor):
        h = self.net(obs)
        mu = self.mu(h)
        log_std = torch.clamp(self.log_std(h), LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

def add_caption(frame: np.ndarray, caption: str, fps: int) -> np.ndarray:
    h, w = frame.shape[:2]
    bar_h = int(h * 0.10)
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    draw.rectangle([(0, h - bar_h), (w, h)], fill=(25, 25, 30))
    draw.rectangle([(0, h - bar_h), (w, h - bar_h + 2)], fill=(255, 255, 255))
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
        except (OSError, IOError):
            font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), caption, font=font)
    tw = bbox[2] - bbox[0]
    tx = (w - tw) // 2
    ty = h - bar_h + (bar_h - (bbox[3] - bbox[1])) // 2 - 2
    draw.text((tx, ty), caption, fill=(220, 220, 230), font=font)
    return np.array(img)

def render_with_style(env, render_mode: str = "rgb_array") -> np.ndarray:
    frame = env.render()
    if frame is None:
        return np.zeros((480, 640, 3), dtype=np.uint8)
    return frame

def record(args):
    device = torch.device("cpu")
    env = gym.make(args.env, render_mode="rgb_array")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    actor = Actor(obs_dim, act_dim, args.hidden, env.action_space.low, env.action_space.high).to(device)
    actor.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    actor.eval()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    best_frames = None
    best_return = -float("inf")
    best_info = ""
    for ep in range(args.episodes):
        obs, info = env.reset()
        frames = []
        done, total_reward = False, 0.0
        while not done:
            frame = render_with_style(env)
            if args.caption:
                ep_caption = f"Episode {ep + 1}/{args.episodes}  |  Reward: {total_reward:.1f}"
                frame = add_caption(frame, ep_caption, args.fps)
            frames.append(frame)
            o = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                mu, _ = actor(o)
                a = (torch.tanh(mu) * actor.action_scale + actor.action_bias).cpu().numpy()[0]
            obs, reward, term, trunc, info = env.step(a)
            total_reward += reward
            done = term or trunc
        print(f"episode {ep + 1}/{args.episodes}: reward={total_reward:.1f}  frames={len(frames)}")
        if total_reward > best_return:
            best_return = total_reward
            best_frames = frames
            best_info = info.get("info", "") if info else ""
    env.close()
    fps = args.fps
    imageio.mimsave(args.out, best_frames, fps=fps, loop=0)
    print(f"saved best episode (reward={best_return:.1f}) -> {args.out}")
    if args.mp4:
        mp4_path = str(Path(args.out).with_suffix(".mp4"))
        imageio.mimsave(mp4_path, best_frames, fps=fps)
        print(f"also saved -> {mp4_path}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="path to actor .pth file")
    p.add_argument("--env", required=True, help="gymnasium env id, e.g. HalfCheetah-v5")
    p.add_argument("--hidden", type=int, default=256, help="actor hidden size used at train time")
    p.add_argument("--out", default="results/agent.gif")
    p.add_argument("--episodes", type=int, default=3, help="record N episodes, keep the best")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--mp4", action="store_true", help="also save an .mp4 alongside the .gif")
    p.add_argument("--caption", action="store_true", help="add caption overlay to frames")
    args = p.parse_args()
    record(args)


if __name__ == "__main__":
    main()
