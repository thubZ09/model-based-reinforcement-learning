from __future__ import annotations
import os
import sys
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dreamerv3.replay import Episode, EpisodicReplay

class TestEpisode:
    def test_empty_episode(self):
        ep = Episode()
        assert len(ep) == 0

    def test_episode_with_data(self):
        ep = Episode()
        ep.obs.append(np.zeros((64, 64, 3)))
        ep.action.append(0.5)
        ep.reward.append(1.0)
        ep.cont.append(1.0)
        ep.is_first.append(1.0)
        assert len(ep) == 1

    def test_episode_multiple_steps(self):
        ep = Episode()
        for _ in range(10):
            ep.obs.append(np.zeros((64, 64, 3)))
            ep.action.append(0.0)
            ep.reward.append(0.0)
            ep.cont.append(1.0)
            ep.is_first.append(0.0)
        assert len(ep) == 10


class TestEpisodicReplay:
    def test_empty_buffer_not_ready(self):
        buf = EpisodicReplay(capacity_steps=100, seq_len=10)
        assert not buf.ready()

    def test_buffer_ready_after_episode(self):
        buf = EpisodicReplay(capacity_steps=100, seq_len=10)
        ep = Episode()
        for _ in range(15):
            ep.obs.append(np.zeros((64, 64, 3)))
            ep.action.append(0.0)
            ep.reward.append(0.0)
            ep.cont.append(1.0)
            ep.is_first.append(0.0 if len(ep) > 1 else 1.0)
        buf.add_episode(ep)
        assert buf.ready()

    def test_buffer_not_ready_short_episode(self):
        buf = EpisodicReplay(capacity_steps=100, seq_len=10)
        ep = Episode()
        for _ in range(5):
            ep.obs.append(np.zeros((64, 64, 3)))
            ep.action.append(0.0)
            ep.reward.append(0.0)
            ep.cont.append(1.0)
            ep.is_first.append(0.0 if len(ep) > 1 else 1.0)
        buf.add_episode(ep)
        assert not buf.ready()

    def test_capacity_eviction(self):
        buf = EpisodicReplay(capacity_steps=20, seq_len=10)
        for i in range(5):
            ep = Episode()
            for _ in range(10):
                ep.obs.append(np.zeros((64, 64, 3)))
                ep.action.append(0.0)
                ep.reward.append(float(i))
                ep.cont.append(1.0)
                ep.is_first.append(0.0 if len(ep) > 1 else 1.0)
            buf.add_episode(ep)
        assert buf.total <= 20

    def test_sample_shape(self):
        buf = EpisodicReplay(capacity_steps=1000, seq_len=10)
        ep = Episode()
        for _ in range(20):
            ep.obs.append(np.random.randint(0, 255, (64, 64, 3)).astype(np.uint8))
            ep.action.append(np.random.randn(2).astype(np.float32))
            ep.reward.append(float(np.random.randn()))
            ep.cont.append(1.0)
            ep.is_first.append(0.0 if len(ep) > 1 else 1.0)
        buf.add_episode(ep)

        device = torch.device("cpu")
        batch = buf.sample(batch_size=4, device=device)

        assert batch["obs"].shape == (4, 10, 3, 64, 64)
        assert batch["action"].shape == (4, 10, 2)
        assert batch["reward"].shape == (4, 10)
        assert batch["cont"].shape == (4, 10)
        assert batch["is_first"].shape == (4, 10)

    def test_sample_is_first(self):
        buf = EpisodicReplay(capacity_steps=1000, seq_len=10)
        ep = Episode()
        for i in range(20):
            ep.obs.append(np.random.randint(0, 255, (64, 64, 3)).astype(np.uint8))
            ep.action.append(np.random.randn(2).astype(np.float32))
            ep.reward.append(float(np.random.randn()))
            ep.cont.append(1.0)
            ep.is_first.append(1.0 if i == 0 else 0.0)
        buf.add_episode(ep)
        device = torch.device("cpu")
        batch = buf.sample(batch_size=4, device=device)
        assert batch["is_first"].sum(dim=1).min() >= 0

    def test_multiple_episodes(self):
        buf = EpisodicReplay(capacity_steps=1000, seq_len=10)
        for _ in range(3):
            ep = Episode()
            for _ in range(15):
                ep.obs.append(np.random.randint(0, 255, (64, 64, 3)).astype(np.uint8))
                ep.action.append(np.random.randn(2).astype(np.float32))
                ep.reward.append(float(np.random.randn()))
                ep.cont.append(1.0)
                ep.is_first.append(0.0 if len(ep) > 1 else 1.0)
            buf.add_episode(ep)
        assert len(buf.episodes) == 3
        assert buf.ready()
