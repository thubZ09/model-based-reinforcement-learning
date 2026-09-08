from __future__ import annotations
import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dreamerv3.networks import (
    ConvDecoder,
    ConvEncoder,
    OneHotST,
    SymlogMSE,
    TwoHotSymlog,
    mlp,
    symexp,
    symlog,
)
class TestSymlogSymexp:
    def test_symlog_positive(self):
        x = torch.tensor([1.0, 10.0, 100.0])
        y = symlog(x)
        assert torch.all(y > 0)
        assert torch.all(torch.diff(y) > 0)

    def test_symlog_negative(self):
        x = torch.tensor([-1.0, -10.0, -100.0])
        y = symlog(x)
        assert torch.all(y < 0)

    def test_symlog_zero(self):
        x = torch.tensor([0.0])
        y = symlog(x)
        assert torch.isclose(y, torch.tensor([0.0]))

    def test_symexp_positive(self):
        x = torch.tensor([1.0, 5.0, 10.0])
        y = symexp(x)
        assert torch.all(y > 0)

    def test_symexp_negative(self):
        x = torch.tensor([-1.0, -5.0, -10.0])
        y = symexp(x)
        assert torch.all(y < 0)

    def test_symlog_symexp_inverse(self):
        x = torch.tensor([0.0, 1.0, 10.0, 100.0, -1.0, -10.0])
        y = symexp(symlog(x))
        assert torch.allclose(y, x, atol=1e-5)

    def test_symexp_symlog_inverse(self):
        x = torch.tensor([0.0, 1.0, 10.0, 50.0, -1.0, -10.0])
        y = symlog(symexp(x))
        assert torch.allclose(y, x, atol=1e-4)

class TestTwoHotSymlog:
    def test_log_prob_shape(self):
        logits = torch.randn(4, 8, 255)
        dist = TwoHotSymlog(logits)
        x = torch.randn(4, 8)
        lp = dist.log_prob(x)
        assert lp.shape == (4, 8)

    def test_mean_shape(self):
        logits = torch.randn(4, 8, 255)
        dist = TwoHotSymlog(logits)
        mean = dist.mean()
        assert mean.shape == (4, 8)

    def test_mean_reasonable_range(self):
        logits = torch.randn(10, 5, 255) * 0.1
        dist = TwoHotSymlog(logits, vmin=-20.0, vmax=20.0)
        mean = dist.mean()
        assert mean.abs().max() <= 20.0 + 1e-3

    def test_log_prob_finite(self):
        logits = torch.randn(2, 3, 100)
        dist = TwoHotSymlog(logits)
        x = torch.tensor([[0.5, -1.0, 3.0], [-2.0, 0.0, 10.0]])
        lp = dist.log_prob(x)
        assert torch.all(torch.isfinite(lp))

class TestSymlogMSE:
    def test_log_prob_finite(self):
        pred = torch.randn(4, 8, 1)
        dist = SymlogMSE(pred)
        target = torch.randn(4, 8, 1)
        lp = dist.log_prob(target)
        assert torch.all(torch.isfinite(lp))

    def test_mean_finite(self):
        pred = torch.randn(4, 8, 1)
        dist = SymlogMSE(pred)
        mean = dist.mean()
        assert torch.all(torch.isfinite(mean))

class TestOneHotST:
    def test_sample_shape(self):
        logits = torch.randn(4, 8, 16, 32)
        dist = OneHotST(logits, uniform_mix=0.01)
        sample = dist.sample()
        assert sample.shape == (4, 8, 16, 32)

    def test_sample_one_hot(self):
        logits = torch.randn(2, 3, 4, 5)
        dist = OneHotST(logits, uniform_mix=0.0)
        sample = dist.sample()
        sums = sample.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_kl_shape(self):
        logits1 = torch.randn(4, 8, 16, 32)
        logits2 = torch.randn(4, 8, 16, 32)
        dist1 = OneHotST(logits1)
        dist2 = OneHotST(logits2)
        kl = dist1.kl(dist2)
        assert kl.shape == (4, 8)

    def test_kl_non_negative(self):
        logits = torch.randn(4, 8, 16, 32)
        dist1 = OneHotST(logits)
        dist2 = OneHotST(logits)
        kl = dist1.kl(dist2)
        assert kl.min() >= -1e-5  

    def test_kl_positive_when_different(self):
        logits1 = torch.randn(4, 8, 16, 32)
        logits2 = torch.zeros_like(logits1)
        dist1 = OneHotST(logits1)
        dist2 = OneHotST(logits2)
        kl = dist1.kl(dist2)
        assert kl.mean() > 0

class TestConvEncoder:
    def test_out_dim(self):
        encoder = ConvEncoder(depth=32)
        assert encoder.out_dim == 8 * 32 * 2 * 2

    def test_out_dim_depth_16(self):
        encoder = ConvEncoder(depth=16)
        assert encoder.out_dim == 8 * 16 * 2 * 2

class TestConvDecoder:
    def test_forward_returns_symlogmse(self):
        decoder = ConvDecoder(feat_dim=1024, depth=16)
        feat = torch.randn(2, 1024)
        out = decoder(feat)
        assert isinstance(out, SymlogMSE)
    def test_output_shape(self):
        decoder = ConvDecoder(feat_dim=1024, depth=16)
        feat = torch.randn(2, 1024)
        out = decoder(feat)
        pred = out.pred
        assert pred.shape == (2, 3, 64, 64)

class TestMLP:
    def test_basic_forward(self):
        net = mlp(in_dim=10, out_dim=5, hidden=32, layers=2)
        x = torch.randn(4, 10)
        y = net(x)
        assert y.shape == (4, 5)

    def test_single_layer(self):
        net = mlp(in_dim=8, out_dim=4, hidden=16, layers=1)
        x = torch.randn(2, 8)
        y = net(x)
        assert y.shape == (2, 4)

    def test_three_layers(self):
        net = mlp(in_dim=8, out_dim=4, hidden=16, layers=3)
        x = torch.randn(2, 8)
        y = net(x)
        assert y.shape == (2, 4)
