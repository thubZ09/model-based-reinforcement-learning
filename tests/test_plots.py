from __future__ import annotations
import os
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from plot_results import parse_log

class TestParseLog:
    def test_step_eval_format(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write("step=   100  eval= -227.9  best= -227.9  t=0.1m\n")
            f.write("step=   200  eval= -150.0  best= -150.0  t=0.2m\n")
            f.write("step=   300  eval= -100.5  best= -100.5  t=0.3m\n")
            fname = f.name
        run = parse_log(Path(fname), "pets", "Pendulum-v1", "PETS")
        os.unlink(fname)
        assert len(run.steps) == 3
        assert run.steps == [100, 200, 300]
        assert run.eval == [-227.9, -150.0, -100.5]
        assert run.best == [-227.9, -150.0, -100.5]

    def test_env_steps_format(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write("env_steps=  5000  eval=  75.4  best=  75.4  train_steps=1000\n")
            f.write("env_steps= 10000  eval= 120.0  best= 120.0  train_steps=2000\n")
            fname = f.name
        run = parse_log(Path(fname), "dreamerv3", "walker", "DreamerV3")
        os.unlink(fname)
        assert len(run.steps) == 2
        assert run.steps == [5000, 10000]
        assert run.eval == [75.4, 120.0]

    def test_empty_log(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write("just some random text\n")
            f.write("no numbers here\n")
            fname = f.name
        run = parse_log(Path(fname), "test", "test-env", "Test")
        os.unlink(fname)
        assert len(run.steps) == 0

    def test_mixed_content(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write("[muzero] device=mps\n")
            f.write("step=   500  eval= -300.0  best= -300.0  t=0.05m\n")
            f.write("some random log line\n")
            f.write("step=  1000  eval= -200.0  best= -200.0  t=0.1m\n")
            fname = f.name
        run = parse_log(Path(fname), "muzero", "CartPole-v1", "MuZero")
        os.unlink(fname)
        assert len(run.steps) == 2
        assert run.steps == [500, 1000]

    def test_negative_values(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write("step=   100  eval= -500.0  best= -500.0\n")
            f.write("step=   200  eval= -450.5  best= -450.5\n")
            fname = f.name
        run = parse_log(Path(fname), "test", "test", "Test")
        os.unlink(fname)
        assert run.eval == [-500.0, -450.5]
        assert run.best == [-500.0, -450.5]

    def test_run_fields(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write("step=   100  eval=  50.0  best=  50.0\n")
            fname = f.name
        run = parse_log(Path(fname), "mbpo", "HalfCheetah-v5", "MBPO")
        os.unlink(fname)
        assert run.algo == "mbpo"
        assert run.env == "HalfCheetah-v5"
        assert run.label == "MBPO"
