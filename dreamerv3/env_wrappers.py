from __future__ import annotations
import numpy as np

class DMCPixelEnv:
    def __init__(
        self,
        domain: str,
        task: str,
        image_size: int = 64,
        action_repeat: int = 2,
        seed: int = 0,
    ):
        from dm_control import suite
        self._env = suite.load(
            domain_name=domain,
            task_name=task,
            task_kwargs={"random": seed},
        )
        self._image_size = image_size
        self._action_repeat = action_repeat
        spec = self._env.action_spec()
        class _AS:
            pass
        self.action_space = _AS()
        self.action_space.low = np.asarray(spec.minimum, dtype=np.float32)
        self.action_space.high = np.asarray(spec.maximum, dtype=np.float32)
        self.action_space.shape = (int(spec.shape[0]),)
        self.observation_shape = (image_size, image_size, 3)

    def _render(self) -> np.ndarray:
        frame = self._env.physics.render(
            height=self._image_size, width=self._image_size, camera_id=0
        )
        return np.ascontiguousarray(frame)
    def reset(self):
        self._env.reset()
        return self._render(), {}
    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        total_reward = 0.0
        truncated = False
        for _ in range(self._action_repeat):
            ts = self._env.step(action)
            total_reward += float(ts.reward or 0.0)
            if ts.last():
                truncated = True
                break
        return self._render(), total_reward, False, truncated, {}
    def sample_action(self) -> np.ndarray:
        return np.random.uniform(
            self.action_space.low, self.action_space.high
        ).astype(np.float32)