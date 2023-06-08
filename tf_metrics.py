from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from absl import logging
from tf_agents.metrics import tf_metric
from tf_agents.replay_buffers import table
from tf_agents.utils import common
from tf_agents.utils import nest_utils


class TFDeque(object):
    def __init__(self, max_len, dtype, shape=(), name="TFDeque"):
        self._max_len = tf.convert_to_tensor(max_len, tf.int32)
        self._spec = tf.TensorSpec(shape, dtype, "Buffer")
        self._buffer = table.Table(self._spec, max_len)
        self._head = common.create_variable(name + "Head", 0, (), tf.int32)

    @property
    def data(self):
        return self._buffer.read(tf.range(self.length))

    @common.function(autograph=True)
    def extend(self, value):
        for v in value:
            self.add(v)

    @common.function(autograph=True)
    def add(self, value):
        position = tf.math.mod(self._head, self._max_len)
        self._buffer.write(position, value)
        self._head.assign_add(1)

    @property
    def length(self):
        return tf.math.minimum(self._head, self._max_len)

    @common.function
    def clear(self):
        self._head.assign(0)

    @common.function(autograph=True)
    def mean(self):
        if tf.math.equal(self._head, 0):
            return tf.zeros(self._spec.shape, self._spec.dtype)
        return tf.math.reduce_mean(self.data, 0)

    @common.function(autograph=True)
    def max(self):
        if tf.math.equal(self._head, 0):
            return tf.fill(self._spec.shape, self._spec.dtype.min)
        return tf.math.reduce_max(self.data, 0)

    @common.function(autograph=True)
    def min(self):
        if tf.math.equal(self._head, 0):
            return tf.fill(self._spec.shape, self._spec.dtype.max)
        return tf.math.reduce_min(self.data, 0)


class EnvironmentSteps(tf_metric.TFStepMetric):
    def __init__(self, name="EnvironmentSteps", prefix="Metrics", dtype=tf.int64):
        super(EnvironmentSteps, self).__init__(name, prefix)
        self.dtype = dtype
        self.environment_steps = common.create_variable("environment_steps", 0, (), self.dtype)

    def call(self, trajectory):
        num_steps = tf.cast(~trajectory.is_boundary(), self.dtype)
        num_steps = tf.math.reduce_sum(num_steps)
        self.environment_steps.assign_add(num_steps)
        return trajectory

    def result(self):
        return tf.identity(self.environment_steps, self.name)

    @common.function
    def reset(self):
        self.environment_steps.assign(0)


class NumberOfEpisodes(tf_metric.TFStepMetric):
    def __init__(self, name="NumberOfEpisodes", prefix="Metrics", dtype=tf.int64):
        super(NumberOfEpisodes, self).__init__(name, prefix)
        self.dtype = dtype
        self.number_episodes = common.create_variable("number_episodes", 0, (), self.dtype)

    def call(self, trajectory):
        num_episodes = tf.cast(trajectory.is_last(), self.dtype)
        num_episodes = tf.math.reduce_sum(num_episodes)
        self.number_episodes.assign_add(num_episodes)
        return trajectory

    def result(self):
        return tf.identity(self.number_episodes, self.name)

    @common.function
    def reset(self):
        self.number_episodes.assign(0)


class AverageReturnMetric(tf_metric.TFStepMetric):
    def __init__(self, name="AverageReturn", prefix="Metrics", dtype=tf.float32, batch_size=1, buffer_size=10):
        super(AverageReturnMetric, self).__init__(name, prefix)
        self._buffer = TFDeque(buffer_size, dtype)
        self._dtype = dtype
        self._return_accumulator = common.create_variable("Accumulator", 0, (batch_size,), dtype)

    @common.function(autograph=True)
    def call(self, trajectory):
        self._return_accumulator.assign(
            tf.where(trajectory.is_first(), tf.zeros_like(self._return_accumulator), self._return_accumulator)
        )
        self._return_accumulator.assign_add(
            tf.math.reduce_sum(trajectory.reward, range(1, len(trajectory.reward.shape)))
        )
        last_episode_indices = tf.squeeze(tf.where(trajectory.is_last()), -1)
        for indx in last_episode_indices:
            self._buffer.add(self._return_accumulator[indx])
        return trajectory

    def result(self):
        return self._buffer.mean()

    @common.function
    def reset(self):
        self._buffer.clear()
        self._return_accumulator.assign(tf.zeros_like(self._return_accumulator))


class MaxReturnMetric(tf_metric.TFStepMetric):
    def __init__(self, name="MaxReturn", prefix="Metrics", dtype=tf.float32, batch_size=1, buffer_size=10):
        super(MaxReturnMetric, self).__init__(name, prefix)
        self._buffer = TFDeque(buffer_size, dtype)
        self._dtype = dtype
        self._return_accumulator = common.create_variable("Accumulator", 0, (batch_size,), dtype)

    @common.function(autograph=True)
    def call(self, trajectory):
        self._return_accumulator.assign(
            tf.where(trajectory.is_first(), tf.zeros_like(self._return_accumulator), self._return_accumulator)
        )
        self._return_accumulator.assign_add(trajectory.reward)
        last_episode_indices = tf.squeeze(tf.where(trajectory.is_last()), -1)
        for indx in last_episode_indices:
            self._buffer.add(self._return_accumulator[indx])
        return trajectory

    def result(self):
        return self._buffer.max()

    @common.function
    def reset(self):
        self._buffer.clear()
        self._return_accumulator.assign(tf.zeros_like(self._return_accumulator))


class MinReturnMetric(tf_metric.TFStepMetric):
    def __init__(self, name="MinReturn", prefix="Metrics", dtype=tf.float32, batch_size=1, buffer_size=10):
        super(MinReturnMetric, self).__init__(name, prefix)
        self._buffer = TFDeque(buffer_size, dtype)
        self._dtype = dtype
        self._return_accumulator = common.create_variable("Accumulator", 0, (batch_size,), dtype)

    @common.function(autograph=True)
    def call(self, trajectory):
        self._return_accumulator.assign(
            tf.where(trajectory.is_first(), tf.zeros_like(self._return_accumulator), self._return_accumulator)
        )
        self._return_accumulator.assign_add(trajectory.reward)
        last_episode_indices = tf.squeeze(tf.where(trajectory.is_last()), -1)
        for indx in last_episode_indices:
            self._buffer.add(self._return_accumulator[indx])
        return trajectory

    def result(self):
        return self._buffer.min()

    @common.function
    def reset(self):
        self._buffer.clear()
        self._return_accumulator.assign(tf.zeros_like(self._return_accumulator))


class AverageEpisodeLengthMetric(tf_metric.TFStepMetric):
    def __init__(self, name="AverageEpisodeLength", prefix="Metrics", dtype=tf.float32, batch_size=1, buffer_size=10):
        super(AverageEpisodeLengthMetric, self).__init__(name, prefix)
        self._buffer = TFDeque(buffer_size, dtype)
        self._dtype = dtype
        self._length_accumulator = common.create_variable("Accumulator", 0, (batch_size,), dtype)

    @common.function(autograph=True)
    def call(self, trajectory):
        non_boundary_indices = tf.squeeze(tf.where(tf.logical_not(trajectory.is_boundary())), -1)
        self._length_accumulator.scatter_add(
            tf.IndexedSlices(tf.ones_like(non_boundary_indices, self._length_accumulator.dtype), non_boundary_indices)
        )
        last_indices = tf.squeeze(tf.where(trajectory.is_last()), -1)
        for indx in last_indices:
            self._buffer.add(self._length_accumulator[indx])
        self._length_accumulator.scatter_update(
            tf.IndexedSlices(tf.zeros_like(last_indices, self._dtype), last_indices)
        )
        return trajectory

    def result(self):
        return self._buffer.mean()

    @common.function
    def reset(self):
        self._buffer.clear()
        self._length_accumulator.assign(tf.zeros_like(self._length_accumulator))


class ChosenActionHistogram(tf_metric.TFHistogramStepMetric):
    def __init__(self, name="ChosenActionHistogram", dtype=tf.int32, buffer_size=100):
        super(ChosenActionHistogram, self).__init__(name)
        self._buffer = TFDeque(buffer_size, dtype)
        self._dtype = dtype

    @common.function
    def call(self, trajectory):
        self._buffer.extend(trajectory.action)
        return trajectory

    @common.function
    def result(self):
        return self._buffer.data

    @common.function
    def reset(self):
        self._buffer.clear()


class AverageReturnMultiMetric(tf_metric.TFMultiMetricStepMetric):
    def __init__(
            self,
            reward_spec,
            name="AverageReturnMultiMetric",
            prefix="Metrics",
            dtype=tf.float32,
            batch_size=1,
            buffer_size=10
    ):
        self._batch_size = batch_size
        self._buffer = tf.nest.map_structure(lambda r: TFDeque(buffer_size, r.dtype, r.shape), reward_spec)
        metric_names = _get_metric_names_from_spec(reward_spec)
        self._dtype = dtype

        def create_acc(spec):
            return common.create_variable(
                "Accumulator/" + spec.name, np.zeros((batch_size,) + spec.shape), (batch_size,) + spec.shape, spec.dtype
            )

        self._return_accumulator = tf.nest.map_structure(create_acc, reward_spec)
        self._reward_spec = reward_spec
        super(AverageReturnMultiMetric, self).__init__(name, prefix, metric_names)

    @common.function(autograph=True)
    def call(self, trajectory):
        nest_utils.assert_same_structure(trajectory.reward, self._reward_spec)
        for buf, return_acc, reward in zip(
                tf.nest.flatten(self._buffer),
                tf.nest.flatten(self._return_accumulator),
                tf.nest.flatten(trajectory.reward)
        ):
            is_start = trajectory.is_first()
            if reward.shape.rank > 1:
                is_start = tf.broadcast_to(tf.reshape(trajectory.is_first(), [-1, 1]), tf.shape(return_acc))
            return_acc.assign(tf.where(is_start, tf.zeros_like(return_acc), return_acc))
            return_acc.assign_add(reward)
            last_episode_indices = tf.squeeze(tf.where(trajectory.is_last()), -1)
            for indx in last_episode_indices:
                buf.add(return_acc[indx])
        return trajectory

    def result(self):
        return tf.nest.map_structure(lambda b: b.mean(), self._buffer)

    @common.function
    def reset(self):
        tf.nest.map_structure(lambda b: b.clear(), self._buffer)
        tf.nest.map_structure(lambda acc: acc.assign(tf.zeros_like(acc)), self._return_accumulator)


def log_metrics(metrics, prefix=''):
    log = [f"{m.name} = {m.result().numpy()}" for m in metrics]
    ntt = "\n\t\t "
    logging.info('%s', f"{prefix} \n\t\t {ntt.join(log)}")


def _get_metric_names_from_spec(reward_spec):
    reward_spec_flat = tf.nest.flatten(reward_spec)
    metric_names_list = tf.nest.map_structure(lambda r: r.name, reward_spec_flat)
    return metric_names_list
