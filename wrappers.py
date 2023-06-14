from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tf_agents.environments import py_environment
from tf_agents.trajectories import time_step as ts


class PyEnvironmentBaseWrapper(py_environment.PyEnvironment):
    def __init__(self, env):
        super(PyEnvironmentBaseWrapper, self).__init__()
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def batched(self):
        return getattr(self._env, "batched", False)

    @property
    def batch_size(self):
        return getattr(self._env, "batch_size", None)

    def _reset(self):
        return self._env.reset()

    def _step(self, action):
        return self._env.step(action)

    def get_info(self):
        return self._env.get_info()

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def close(self):
        return self._env.close()

    def render(self):
        return self._env.render()

    def seed(self, seed):
        return self._env.seed(seed)

    def wrapped_env(self):
        return self._env

    def set_state(self, state):
        self._env.set_state(state)

    def get_state(self):
        return self._env.get_state()


class TimeLimit(PyEnvironmentBaseWrapper):
    def __init__(self, env, duration):
        super(TimeLimit, self).__init__(env)
        self._duration = duration
        self._num_steps = None

    def _reset(self):
        self._num_steps = 0
        return self._env.reset()

    def _step(self, action):
        if self._num_steps is None:
            return self.reset()
        time_step = self._env.step(action)
        self._num_steps += 1
        if self._num_steps >= self._duration:
            time_step = time_step._replace(step_type=ts.StepType.LAST)
        if time_step.is_last():
            self._num_steps = None
        return time_step

    @property
    def duration(self):
        return self._duration
