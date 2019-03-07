"""Util functions.
"""

import tensorflow as tf
import numpy as np

def write_summary(value, tag, summary_writer, global_step):
  """Write a single summary value to tensorboard"""
  summary = tf.Summary()
  summary.value.add(tag=tag, simple_value=value)
  summary_writer.add_summary(summary, global_step)


def exp_decay_with_warmup(global_step,
                             learning_rate_base,
                             total_steps,
                             warmup_learning_rate=0.0,
                             warmup_steps=0,
                             hold_base_rate_steps=0):
  """Exponential decay schedule with warm up period.
  Cosine annealing learning rate as described in:
    Loshchilov and Hutter, SGDR: Stochastic Gradient Descent with Warm Restarts.
    ICLR 2017. https://arxiv.org/abs/1608.03983
  In this schedule, the learning rate grows linearly from warmup_learning_rate
  to learning_rate_base for warmup_steps, then transitions to a cosine decay
  schedule.
  Args:
    global_step: int64 (scalar) tensor representing global step.
    learning_rate_base: base learning rate.
    total_steps: total number of training steps.
    warmup_learning_rate: initial learning rate for warm up.
    warmup_steps: number of warmup steps.
    hold_base_rate_steps: Optional number of steps to hold base learning rate
      before decaying.
  Returns:
    a (scalar) float tensor representing learning rate.
  Raises:
    ValueError: if warmup_learning_rate is larger than learning_rate_base,
      or if warmup_steps is larger than total_steps.
  """
  if total_steps < warmup_steps:
    raise ValueError('total_steps must be larger or equal to '
                     'warmup_steps.')
  learning_rate = 0.5 * learning_rate_base * (1 + tf.cos(
      np.pi *
      (tf.cast(global_step, tf.float32) - warmup_steps - hold_base_rate_steps
      ) / float(total_steps - warmup_steps - hold_base_rate_steps)))
  if hold_base_rate_steps > 0:
    learning_rate = tf.where(global_step > warmup_steps + hold_base_rate_steps,
                             learning_rate, learning_rate_base)
  if warmup_steps > 0:
    if learning_rate_base < warmup_learning_rate:
      raise ValueError('learning_rate_base must be larger or equal to '
                       'warmup_learning_rate.')
    slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
    warmup_rate = slope * tf.cast(global_step,
                                  tf.float32) + warmup_learning_rate
    learning_rate = tf.where(global_step < warmup_steps, warmup_rate,
                             learning_rate)
  return tf.where(global_step > total_steps, 0.0, learning_rate,
                  name='learning_rate')

def cosine_decay_with_warmup(global_step,
                             learning_rate_base,
                             total_steps,
                             warmup_learning_rate=0.0,
                             warmup_steps=0,
                             hold_base_rate_steps=0):
  """Cosine decay schedule with warm up period.
  Cosine annealing learning rate as described in:
    Loshchilov and Hutter, SGDR: Stochastic Gradient Descent with Warm Restarts.
    ICLR 2017. https://arxiv.org/abs/1608.03983
  In this schedule, the learning rate grows linearly from warmup_learning_rate
  to learning_rate_base for warmup_steps, then transitions to a cosine decay
  schedule.
  Args:
    global_step: int64 (scalar) tensor representing global step.
    learning_rate_base: base learning rate.
    total_steps: total number of training steps.
    warmup_learning_rate: initial learning rate for warm up.
    warmup_steps: number of warmup steps.
    hold_base_rate_steps: Optional number of steps to hold base learning rate
      before decaying.
  Returns:
    a (scalar) float tensor representing learning rate.
  Raises:
    ValueError: if warmup_learning_rate is larger than learning_rate_base,
      or if warmup_steps is larger than total_steps.
  """
  if total_steps < warmup_steps:
    raise ValueError('total_steps must be larger or equal to '
                     'warmup_steps.')
  learning_rate = 0.5 * learning_rate_base * (1 + tf.cos(
      np.pi *
      (tf.cast(global_step, tf.float32) - warmup_steps - hold_base_rate_steps
      ) / float(total_steps - warmup_steps - hold_base_rate_steps)))
  if hold_base_rate_steps > 0:
    learning_rate = tf.where(global_step > warmup_steps + hold_base_rate_steps,
                             learning_rate, learning_rate_base)
  if warmup_steps > 0:
    if learning_rate_base < warmup_learning_rate:
      raise ValueError('learning_rate_base must be larger or equal to '
                       'warmup_learning_rate.')
    slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
    warmup_rate = slope * tf.cast(global_step,
                                  tf.float32) + warmup_learning_rate
    learning_rate = tf.where(global_step < warmup_steps, warmup_rate,
                             learning_rate)
  return tf.where(global_step > total_steps, 0.0, learning_rate,
                  name='learning_rate')
