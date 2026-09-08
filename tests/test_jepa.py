from __future__ import annotations
import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from jepa_nbody.model import Decoder, Encoder, JEPANBody, Predictor, variance_reg
from jepa_nbody.simulate import (
    accelerations,
    generate,
    total_energy,
    velocity_verlet_step,
)

class TestAccelerations:
    def test_shape(self):
        pos = torch.randn(2, 4, 2)
        mass = torch.randn(2, 4).abs()
        acc = accelerations(pos, mass)
        assert acc.shape == (2, 4, 2)
    def test_symmetry(self):
        torch.manual_seed(42)
        pos = torch.randn(1, 3, 2)
        mass = torch.rand(1, 3) + 0.1
        acc = accelerations(pos, mass)
        total_acc = (mass.unsqueeze(-1) * acc).sum(dim=-2)
        assert total_acc.abs().max() < 10.0 
    def test_finite(self):
        pos = torch.randn(1, 4, 2)
        mass = torch.rand(1, 4) + 0.1
        acc = accelerations(pos, mass)
        assert torch.all(torch.isfinite(acc))

class TestVelocityVerlet:
    def test_shape(self):
        pos = torch.randn(1, 4, 2)
        vel = torch.randn(1, 4, 2)
        mass = torch.rand(1, 4) + 0.1
        pos_new, vel_new = velocity_verlet_step(pos, vel, mass, dt=0.01)
        assert pos_new.shape == pos.shape
        assert vel_new.shape == vel.shape
    def test_finite(self):
        pos = torch.randn(1, 4, 2)
        vel = torch.randn(1, 4, 2)
        mass = torch.rand(1, 4) + 0.1
        pos_new, vel_new = velocity_verlet_step(pos, vel, mass, dt=0.01)
        assert torch.all(torch.isfinite(pos_new))
        assert torch.all(torch.isfinite(vel_new))

class TestTotalEnergy:
    def test_shape(self):
        pos = torch.randn(2, 4, 2)
        vel = torch.randn(2, 4, 2)
        mass = torch.rand(2, 4) + 0.1
        e = total_energy(pos, vel, mass)
        assert e.shape == (2,)
    def test_single_trajectory(self):
        pos = torch.randn(1, 4, 2)
        vel = torch.randn(1, 4, 2)
        mass = torch.rand(1, 4) + 0.1
        e = total_energy(pos, vel, mass)
        assert e.shape == (1,)
    def test_finite(self):
        pos = torch.randn(1, 4, 2)
        vel = torch.randn(1, 4, 2)
        mass = torch.rand(1, 4) + 0.1
        e = total_energy(pos, vel, mass)
        assert torch.all(torch.isfinite(e))

class TestGenerate:
    def test_shape(self):
        pos, vel, mass = generate(n_traj=4, n_frames=10, n_bodies=4, seed=42)
        assert pos.shape == (4, 10, 4, 2)
        assert vel.shape == (4, 10, 4, 2)
        assert mass.shape == (4, 4)
    def test_reproducible(self):
        pos1, vel1, mass1 = generate(n_traj=2, n_frames=5, n_bodies=3, seed=123)
        pos2, vel2, mass2 = generate(n_traj=2, n_frames=5, n_bodies=3, seed=123)
        assert torch.allclose(pos1, pos2)
        assert torch.allclose(vel1, vel2)
        assert torch.allclose(mass1, mass2)
    def test_different_seeds(self):
        pos1, _, _ = generate(n_traj=2, n_frames=5, n_bodies=3, seed=1)
        pos2, _, _ = generate(n_traj=2, n_frames=5, n_bodies=3, seed=2)
        assert not torch.allclose(pos1, pos2)

class TestEncoder:
    def test_forward(self):
        enc = Encoder(state_dim=16, latent_dim=32)
        x = torch.randn(4, 16)
        z = enc(x)
        assert z.shape == (4, 32)

class TestPredictor:
    def test_forward(self):
        pred = Predictor(latent_dim=32, n_bodies=4)
        z = torch.randn(4, 32)
        mass = torch.rand(4, 4)
        z_out = pred(z, mass)
        assert z_out.shape == (4, 32)

class TestDecoder:
    def test_forward(self):
        dec = Decoder(latent_dim=32, state_dim=16)
        z = torch.randn(4, 32)
        x = dec(z)
        assert x.shape == (4, 16)

class TestJEPANBody:
    def test_forward(self):
        model = JEPANBody(state_dim=16, n_bodies=4, latent_dim=32)
        state = torch.randn(2, 16)
        mass = torch.rand(2, 4)
        z = model.encoder(state)
        z_pred = model.predictor(z, mass)
        recon = model.decoder(z_pred)
        assert z.shape == (2, 32)
        assert z_pred.shape == (2, 32)
        assert recon.shape == (2, 16)

    def test_update_target(self):
        model = JEPANBody(state_dim=16, n_bodies=4, latent_dim=32, ema_tau=0.99)
        old_state = {k: v.clone() for k, v in model.target_encoder.state_dict().items()}
        model.update_target()
        for k in old_state:
            new_state = dict(model.target_encoder.state_dict())[k]
            assert new_state.shape == old_state[k].shape

    def test_target_encoder_no_grad(self):
        model = JEPANBody(state_dim=16, n_bodies=4, latent_dim=32)
        for p in model.target_encoder.parameters():
            assert not p.requires_grad

class TestVarianceReg:
    def test_shape(self):
        z = torch.randn(4, 32)
        vr = variance_reg(z)
        assert vr.dim() == 0 
    def test_shape_batched(self):
        z = torch.randn(2, 4, 32)
        vr = variance_reg(z)
        assert vr.dim() == 0
    def test_non_negative(self):
        z = torch.randn(100, 64) * 2.0 
        vr = variance_reg(z)
        assert vr >= 0  
    def test_positive_when_low_variance(self):
        z = torch.randn(100, 64) * 0.1 
        vr = variance_reg(z)
        assert vr > 0
