from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium.error import DependencyNotInstalled
from gymnasium.spaces import Box


class LazyFrames:
    __slots__ = ("frame_shape", "dtype", "shape", "lz4_compress", "_frames")

    def __init__(self, frames, lz4_compress=False):
        self.frame_shape = frames[0].shape,
        self.shape = self.frame_shape, len(frames)
        self.dtype = frames[0].dtype
        if lz4_compress:
            try:
                from lz4.block import compress
            except ImportError as e:
                raise DependencyNotInstalled("lz4 is not installed, run `pip install gymnasium[other]`") from e
            frames = [compress(frame) for frame in frames]
        self._frames = frames
        self.lz4_compress = lz4_compress

    def __array__(self, dtype=None):
        arr = self[:]
        if dtype is not None:
            return arr.astype(dtype)
        return arr

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, int_or_slice):
        if isinstance(int_or_slice, int):
            return self._check_decompress(self._frames[int_or_slice])
        return np.stack([self._check_decompress(f) for f in self._frames[int_or_slice]], -1)

    def __eq__(self, other):
        return self.__array__() == other

    def _check_decompress(self, frame):
        if self.lz4_compress:
            from lz4.block import decompress
            return np.frombuffer(decompress(frame), self.dtype).reshape(self.frame_shape)
        return frame


class FrameStack(gym.ObservationWrapper, gym.utils.RecordConstructorArgs):
    def __init__(self, env, num_stack, lz4_compress=False):
        gym.utils.RecordConstructorArgs.__init__(self, num_stack=num_stack, lz4_compress=lz4_compress)
        gym.ObservationWrapper.__init__(self, env)
        self.num_stack = num_stack
        self.lz4_compress = lz4_compress
        self.frames = deque(maxlen=num_stack)
        low = np.repeat(self.observation_space.low[..., np.newaxis], num_stack, -1)
        high = np.repeat(self.observation_space.high[..., np.newaxis], num_stack, -1)
        self.observation_space = Box(low, high, dtype=self.observation_space.dtype)

    def observation(self):
        assert len(self.frames) == self.num_stack, (len(self.frames), self.num_stack)
        return LazyFrames(list(self.frames), self.lz4_compress)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(observation)
        return self.observation(), reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        [self.frames.append(obs) for _ in range(self.num_stack)]
        return self.observation(), info
