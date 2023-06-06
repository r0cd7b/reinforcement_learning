from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Callable

import gin
import gymnasium as gym
from tf_agents.environments import py_environment
from tf_agents.environments import wrappers

import gym_wrapper

TimeLimitWrapperType = Callable[[py_environment.PyEnvironment, int], py_environment.PyEnvironment]


@gin.configurable
def load(
        environment_name,
        discount=1.0,
        max_episode_steps=None,
        gym_env_wrappers=(),
        env_wrappers=(),
        spec_dtype_map=None,
        gym_kwargs=None
):
    gym_kwargs = gym_kwargs if gym_kwargs else {}
    gym_spec = gym.spec(environment_name)
    gym_env = gym_spec.make(**gym_kwargs)
    if max_episode_steps is None and gym_spec.max_episode_steps is not None:
        max_episode_steps = gym_spec.max_episode_steps
    return wrap_env(
        gym_env, discount, max_episode_steps, gym_env_wrappers, env_wrappers=env_wrappers, spec_dtype_map=spec_dtype_map
    )


@gin.configurable
def wrap_env(
        gym_env,
        discount=1.0,
        max_episode_steps=None,
        gym_env_wrappers=(),
        time_limit_wrapper=wrappers.TimeLimit,
        env_wrappers=(),
        spec_dtype_map=None,
        auto_reset=True
):
    for wrapper in gym_env_wrappers:
        gym_env = wrapper(gym_env)
    env = gym_wrapper.GymWrapper(gym_env, discount, spec_dtype_map, auto_reset=auto_reset)
    if max_episode_steps is not None and max_episode_steps > 0:
        env = time_limit_wrapper(env, max_episode_steps)
    for wrapper in env_wrappers:
        env = wrapper(env)
    return env
