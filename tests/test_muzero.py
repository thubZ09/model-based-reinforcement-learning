from __future__ import annotations
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from muzero_cartpole import (
    MinMaxStats,
    MuZeroNet,
    Node,
    Trajectory,
    TrajectoryBuffer,
    expand_node,
    scalar_to_support,
    select_action,
    select_child,
    support_to_scalar,
    temperature,
    ucb_score,
)

class TestScalarSupport:
    def test_positive_scalar(self):
        x = torch.tensor([5.0])
        support = scalar_to_support(x, support_size=25)
        assert support.shape == (1, 51)
        assert torch.all(torch.isfinite(support))
    def test_negative_scalar(self):
        x = torch.tensor([-3.0])
        support = scalar_to_support(x, support_size=25)
        assert support.shape == (1, 51)
    def test_zero_scalar(self):
        x = torch.tensor([0.0])
        support = scalar_to_support(x, support_size=25)
        assert support.shape == (1, 51)
    def test_batch(self):
        x = torch.tensor([[1.0, -2.0], [3.0, -4.0]])
        support = scalar_to_support(x, support_size=25)
        assert support.shape == (2, 2, 51)

class TestSupportToScalar:
    def test_basic(self):
        support_size = 25
        support = torch.zeros(1, 2 * support_size + 1)
        support[0, support_size + 5] = 1.0  # peak at +5
        x = support_to_scalar(support.log(), support_size)
        assert torch.isfinite(x)
        assert x.item() > 0
    def test_negative(self):
        support_size = 25
        support = torch.zeros(1, 2 * support_size + 1)
        support[0, support_size - 3] = 1.0  # peak at -3
        x = support_to_scalar(support.log(), support_size)
        assert torch.isfinite(x)
        assert x.item() < 0

class TestNode:
    def test_new_node(self):
        node = Node(prior=0.5)
        assert node.prior == 0.5
        assert node.visit_count == 0
        assert node.value() == 0.0
        assert not node.expanded()
    def test_value_after_visits(self):
        node = Node(prior=0.5)
        node.value_sum = 10.0
        node.visit_count = 2
        assert node.value() == 5.0
    def test_value_zero_visits(self):
        node = Node(prior=0.5)
        assert node.value() == 0.0

class TestMinMaxStats:
    def test_normalize_default(self):
        mm = MinMaxStats()
        assert mm.normalize(5.0) == 5.0
    def test_normalize_after_update(self):
        mm = MinMaxStats()
        mm.update(0.0)
        mm.update(10.0)
        assert mm.normalize(5.0) == 0.5
        assert mm.normalize(0.0) == 0.0
        assert mm.normalize(10.0) == 1.0

class TestUCB:
    def test_ucb_score_computes(self):
        parent = Node(prior=1.0)
        parent.visit_count = 10
        child = Node(prior=0.5)
        child.visit_count = 3
        child.value_sum = 6.0
        child.reward = 1.0
        child.cont = 1.0
        mm = MinMaxStats()
        mm.update(1.0)
        score = ucb_score(
            parent,
            child,
            mm,
            type("Args", (), {"pb_c_base": 19652.0, "pb_c_init": 1.25, "discount": 0.997})(),
        )
        assert score > 0

class TestSelectChild:
    def test_selects_max_ucb(self):
        node = Node(prior=1.0)
        node.visit_count = 10
        child_a = Node(prior=0.9)
        child_a.visit_count = 1
        child_a.value_sum = 10.0
        child_a.reward = 5.0
        child_a.cont = 1.0
        child_b = Node(prior=0.1)
        child_b.visit_count = 5
        child_b.value_sum = 5.0
        child_b.reward = 1.0
        child_b.cont = 1.0
        node.children[0] = child_a
        node.children[1] = child_b
        mm = MinMaxStats()
        mm.update(5.0)
        a_chosen, _ = select_child(
            node,
            mm,
            type("Args", (), {"pb_c_base": 19652.0, "pb_c_init": 1.25, "discount": 0.997})(),
        )
        assert a_chosen == 0

class TestExpandNode:
    def test_expands_with_actions(self):
        hidden = torch.randn(1, 64)
        policy_logits = torch.randn(2)  
        node = Node(prior=1.0)
        expand_node(node, hidden, 0.0, 1.0, policy_logits, 2)
        assert len(node.children) == 2

class TestTrajectory:
    def test_empty(self):
        traj = Trajectory()
        assert len(traj) == 0
    def test_with_data(self):
        traj = Trajectory()
        traj.actions.append(0)
        traj.obs.append(np.zeros(4))
        traj.rewards.append(1.0)
        traj.policies.append(np.array([0.5, 0.5]))
        traj.values.append(0.0)
        traj.dones.append(False)
        assert len(traj) == 1

class TestTrajectoryBuffer:
    def test_empty_sample(self):
        buf = TrajectoryBuffer(max_traj=10, max_transitions=1000)
        assert len(buf.trajectories) == 0
    def test_add_and_sample(self):
        buf = TrajectoryBuffer(max_traj=10, max_transitions=1000)
        traj = Trajectory()
        for _ in range(20):
            traj.actions.append(0)
            traj.obs.append(np.zeros(4))
            traj.rewards.append(1.0)
            traj.policies.append(np.array([0.5, 0.5]))
            traj.values.append(0.0)
            traj.dones.append(False)
        buf.add(traj)
        assert len(buf.trajectories) == 1
    def test_max_transitions_eviction(self):
        buf = TrajectoryBuffer(max_traj=100, max_transitions=30)
        for _ in range(5):
            traj = Trajectory()
            for _ in range(10):
                traj.actions.append(0)
                traj.obs.append(np.zeros(4))
                traj.rewards.append(1.0)
                traj.policies.append(np.array([0.5, 0.5]))
                traj.values.append(0.0)
                traj.dones.append(False)
            buf.add(traj)
        assert buf.total <= 30

class TestTemperature:
    def test_initial_temperature(self):
        args = type(
            "Args",
            (),
            {
                "temperature_init": 1.0,
                "temperature_final": 0.25,
                "temperature_decay_steps": 1000,
            },
        )()
        assert temperature(0, args) == 1.0
    def test_final_temperature(self):
        args = type(
            "Args",
            (),
            {
                "temperature_init": 1.0,
                "temperature_final": 0.25,
                "temperature_decay_steps": 1000,
            },
        )()
        assert temperature(1000, args) == 0.25
        assert temperature(2000, args) == 0.25
    def test_decay(self):
        args = type(
            "Args", (), {"temperature_init": 1.0, "temperature_final": 0.0, "temperature_decay_steps": 100}
        )()
        t = temperature(50, args)
        assert 0.0 < t < 1.0

class TestSelectAction:
    def test_zero_visits(self):
        root = Node(prior=1.0)
        root.children[0] = Node(prior=0.5)
        root.children[1] = Node(prior=0.5)
        action = select_action(root, temp=1.0, num_actions=2)
        assert action in [0, 1]
    def test_zero_temperature_argmax(self):
        root = Node(prior=1.0)
        root.children[0] = Node(prior=0.5)
        root.children[0].visit_count = 100
        root.children[1] = Node(prior=0.5)
        root.children[1].visit_count = 1
        action = select_action(root, temp=0.0, num_actions=2)
        assert action == 0
    def test_high_temperature_random(self):
        root = Node(prior=1.0)
        root.children[0] = Node(prior=0.5)
        root.children[0].visit_count = 50
        root.children[1] = Node(prior=0.5)
        root.children[1].visit_count = 50
        actions = [select_action(root, temp=10.0, num_actions=2) for _ in range(100)]
        assert 0 in actions or 1 in actions

class TestMuZeroNet:
    def test_initial(self):
        args = type(
            "Args",
            (),
            {
                "hidden_dim": 64,
                "support_size": 25,
                "repr_layers": 2,
                "dynamics_layers": 2,
                "prediction_layers": 2,
            },
        )()
        net = MuZeroNet(obs_dim=4, num_actions=2, args=args)
        obs = torch.randn(1, 4)
        hidden, policy, value = net.initial(obs)
        assert hidden.shape == (1, 64)
        assert policy.shape == (1, 2)
        assert value.shape == (1, 51)

    def test_recurrent(self):
        args = type(
            "Args",
            (),
            {
                "hidden_dim": 64,
                "support_size": 25,
                "repr_layers": 2,
                "dynamics_layers": 2,
                "prediction_layers": 2,
            },
        )()
        net = MuZeroNet(obs_dim=4, num_actions=2, args=args)
        obs = torch.randn(1, 4)
        hidden, _, _ = net.initial(obs)
        action = torch.tensor([0])
        s_next, _r_logits, _c_logit, p_logits, v_logits = net.recurrent(hidden, action)
        assert s_next.shape == (1, 64)
        assert p_logits.shape == (1, 2)
        assert v_logits.shape == (1, 51)
