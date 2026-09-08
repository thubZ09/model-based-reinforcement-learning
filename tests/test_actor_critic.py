from __future__ import annotations
import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dreamerv3.actor_critic import (
    ActorCriticConfig,
    Critic,
    ReturnNormalizer,
    TanhNormalActor,
    ema_update,
    lambda_return,
)

class TestTanhNormalActor:
    def test_dist_shape(self):
        actor = TanhNormalActor(feat_dim=64, act_dim=3, action_low=-1, action_high=1)
        feat = torch.randn(4, 64)
        d = actor.dist(feat)
        assert d.mean.shape == (4, 3)
        assert d.stddev.shape == (4, 3)
    def test_sample_shape(self):
        actor = TanhNormalActor(feat_dim=64, act_dim=3, action_low=-1, action_high=1)
        feat = torch.randn(4, 64)
        action, logp, ent = actor.sample(feat)
        assert action.shape == (4, 3)
        assert logp.shape == (4,)
        assert ent.shape == (4,)
    def test_sample_in_range(self):
        actor = TanhNormalActor(feat_dim=64, act_dim=3, action_low=-2.0, action_high=2.0)
        feat = torch.randn(100, 64)
        action, _, _ = actor.sample(feat)
        assert (action >= -2.0 - 1e-5).all()
        assert (action <= 2.0 + 1e-5).all()
    def test_deterministic_shape(self):
        actor = TanhNormalActor(feat_dim=64, act_dim=3, action_low=-1, action_high=1)
        feat = torch.randn(4, 64)
        action = actor.deterministic(feat)
        assert action.shape == (4, 3)
    def test_deterministic_in_range(self):
        actor = TanhNormalActor(feat_dim=64, act_dim=3, action_low=-2.0, action_high=2.0)
        feat = torch.randn(100, 64)
        action = actor.deterministic(feat)
        assert (action >= -2.0 - 1e-5).all()
        assert (action <= 2.0 + 1e-5).all()

class TestCritic:
    def test_dist_shape(self):
        cfg = ActorCriticConfig(num_bins=255)
        critic = Critic(feat_dim=64, cfg=cfg)
        feat = torch.randn(4, 64)
        dist = critic.dist(feat)
        assert dist.logits.shape == (4, 255)
    def test_mean_shape(self):
        cfg = ActorCriticConfig(num_bins=255)
        critic = Critic(feat_dim=64, cfg=cfg)
        feat = torch.randn(4, 64)
        mean = critic.dist(feat).mean()
        assert mean.shape == (4,)

class TestLambdaReturn:
    def test_shape(self):
        rewards = torch.randn(10)
        values = torch.randn(11)
        continues = torch.ones(11)
        ret = lambda_return(rewards, values, continues, gamma=0.99, lambda_=0.95)
        assert ret.shape == (10,)
    def test_all_continues_one(self):
        rewards = torch.tensor([1.0, 2.0, 3.0])
        values = torch.tensor([0.0, 1.0, 2.0, 3.0])
        continues = torch.ones(4)
        ret = lambda_return(rewards, values, continues, gamma=1.0, lambda_=1.0)
        assert ret.shape == (3,)
        assert torch.all(torch.isfinite(ret))
    def test_discounted_no_bootstrap(self):
        rewards = torch.tensor([1.0, 2.0, 3.0])
        values = torch.zeros(4)
        continues = torch.tensor([1.0, 1.0, 1.0, 0.0])
        ret = lambda_return(rewards, values, continues, gamma=0.9, lambda_=0.0)
        assert ret.shape == (3,)

    def test_finite(self):
        rewards = torch.randn(20)
        values = torch.randn(21)
        continues = torch.rand(21)
        ret = lambda_return(rewards, values, continues, gamma=0.99, lambda_=0.95)
        assert torch.all(torch.isfinite(ret))

class TestReturnNormalizer:
    def test_initial_std(self):
        norm = ReturnNormalizer(decay=0.99, limit=1.0)
        assert norm.std == 1.0
        assert not norm.initialized
    def test_update_initializes(self):
        norm = ReturnNormalizer(decay=0.99, limit=1.0)
        returns = torch.tensor([10.0, 20.0, 30.0])
        norm.update(returns)
        assert norm.initialized
        assert norm.std > 0
    def test_scale_returns_positive(self):
        norm = ReturnNormalizer(decay=0.99, limit=1.0)
        returns = torch.tensor([10.0, 20.0, 30.0])
        norm.update(returns)
        scale = norm.scale()
        assert scale >= 1.0
    def test_scale_with_limit(self):
        norm = ReturnNormalizer(decay=0.99, limit=5.0)
        returns = torch.tensor([1.0, 1.1, 0.9])
        norm.update(returns)
        scale = norm.scale()
        assert scale >= 5.0
    def test_multiple_updates(self):
        norm = ReturnNormalizer(decay=0.99, limit=0.1)
        norm.update(torch.tensor([10.0]))
        norm.update(torch.tensor([10.1]))
        assert norm.std > 0

class TestEMAUpdate:
    def test_ema_same_initial(self):
        source = torch.nn.Linear(10, 5)
        target = torch.nn.Linear(10, 5)
        target.load_state_dict(source.state_dict())
        ema_update(target, source, tau=0.02)
        for p_t, p_s in zip(target.parameters(), source.parameters(), strict=True):
            diff = (p_t.data - p_s.data).abs().max()
            assert diff < 0.02 * 2 
    def test_ema_full_copy_tau_1(self):
        source = torch.nn.Linear(10, 5)
        target = torch.nn.Linear(10, 5)
        target.load_state_dict(source.state_dict())
        ema_update(target, source, tau=1.0)
        for p_t, p_s in zip(target.parameters(), source.parameters(), strict=True):
            assert torch.allclose(p_t.data, p_s.data)
