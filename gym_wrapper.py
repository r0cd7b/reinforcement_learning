from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest
from tf_agents import specs
from tf_agents.environments import py_environment
from tf_agents.trajectories import time_step as ts


def spec_from_gym_space(
        space,
        dtype_map=None,
        simplify_box_bounds=True,
        name=None
):
    if dtype_map is None:
        dtype_map = {}

    def try_simplify_array_to_value(np_array):
        first_value = np_array.item(0)
        if np.all(np_array == first_value):
            return np.array(first_value, np_array.dtype)
        else:
            return np_array

    def nested_spec(spec, child_name):
        nested_name = name + '/' + child_name if name else child_name
        return spec_from_gym_space(spec, dtype_map, simplify_box_bounds, nested_name)

    if isinstance(space, gym.spaces.Discrete):
        maximum = space.n - 1
        dtype = dtype_map.get(gym.spaces.Discrete, np.int64)
        return specs.BoundedArraySpec((), dtype, 0, maximum, name)
    elif isinstance(space, gym.spaces.MultiDiscrete):
        dtype = dtype_map.get(gym.spaces.MultiDiscrete, np.int32)
        maximum = try_simplify_array_to_value(np.asarray(space.nvec - 1, dtype))
        return specs.BoundedArraySpec(space.shape, dtype, 0, maximum, name)
    elif isinstance(space, gym.spaces.MultiBinary):
        dtype = dtype_map.get(gym.spaces.MultiBinary, np.int32)
        if isinstance(space.n, int):
            shape = (space.n,)
        else:
            shape = tuple(space.n)
        return specs.BoundedArraySpec(shape, dtype, 0, 1, name)
    elif isinstance(space, gym.spaces.Box):
        if hasattr(space, "dtype") and gym.spaces.Box not in dtype_map:
            dtype = space.dtype
        else:
            dtype = dtype_map.get(gym.spaces.Box, np.float32)
        if dtype == tf.string:
            return specs.ArraySpec(space.shape, dtype, name)
        minimum = np.asarray(space.low, dtype)
        maximum = np.asarray(space.high, dtype)
        if simplify_box_bounds:
            simple_minimum = try_simplify_array_to_value(minimum)
            simple_maximum = try_simplify_array_to_value(maximum)
            if simple_minimum.shape == simple_maximum.shape:
                minimum = simple_minimum
                maximum = simple_maximum
        return specs.BoundedArraySpec(space.shape, dtype, minimum, maximum, name)
    elif isinstance(space, gym.spaces.Tuple):
        return tuple(
            [nested_spec(s, f"tuple_{i}") for i, s in enumerate(space.spaces)])
    elif isinstance(space, gym.spaces.Dict):
        return collections.OrderedDict([(key, nested_spec(s, key)) for key, s in space.spaces.items()])
    else:
        raise ValueError(f"The gym space {space} is currently not supported.")


class GymWrapper(py_environment.PyEnvironment):
    def __init__(
            self,
            gym_env,
            discount=1.0,
            spec_dtype_map=None,
            match_obs_space_dtype=True,
            auto_reset=True,
            simplify_box_bounds=True
    ):
        super(GymWrapper, self).__init__(auto_reset)
        self._gym_env = gym_env
        self._discount = discount
        self._action_is_discrete = isinstance(self._gym_env.action_space, gym.spaces.Discrete)
        self._match_obs_space_dtype = match_obs_space_dtype
        self._observation_spec = spec_from_gym_space(
            self._gym_env.observation_space, spec_dtype_map, simplify_box_bounds, "observation"
        )
        self._action_spec = spec_from_gym_space(
            self._gym_env.action_space, spec_dtype_map, simplify_box_bounds, "action"
        )
        self._flat_obs_spec = tf.nest.flatten(self._observation_spec)
        self._info = None
        self._done = True

    @property
    def gym(self):
        return self._gym_env

    def __getattr__(self, name):
        gym_env = super(GymWrapper, self).__getattribute__("_gym_env")
        return getattr(gym_env, name)

    def get_info(self):
        return self._info

    def _reset(self):
        observation, self._info = self._gym_env.reset()
        self._done = False
        if self._match_obs_space_dtype:
            observation = self._to_obs_space_dtype(observation)
        return ts.restart(observation)

    @property
    def done(self):
        return self._done

    def _step(self, action):
        if self._action_is_discrete and isinstance(action, np.ndarray):
            action = action.item()
        observation, reward, terminated, truncated, self._info = self._gym_env.step(action)
        self._done = terminated | truncated
        if self._match_obs_space_dtype:
            observation = self._to_obs_space_dtype(observation)
        if self._done:
            return ts.termination(observation, reward)
        else:
            return ts.transition(observation, reward, self._discount)

    def _to_obs_space_dtype(self, observation):
        flat_obs = nest.flatten_up_to(self._observation_spec, observation)
        matched_observations = []
        for spec, obs in zip(self._flat_obs_spec, flat_obs):
            matched_observations.append(np.asarray(obs, spec.dtype))
        return tf.nest.pack_sequence_as(self._observation_spec, matched_observations)

    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._action_spec

    def close(self):
        return self._gym_env.close()

    def seed(self, seed):
        seed_value = self._gym_env.seed(seed)
        if seed_value is None:
            return 0
        return seed_value

    def render(self):
        return self._gym_env.render()

    def set_state(self, state):
        return self._gym_env.set_state(state)

    def get_state(self):
        return self._gym.get_state()
