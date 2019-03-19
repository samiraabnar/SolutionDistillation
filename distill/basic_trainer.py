import tensorflow as tf
import numpy as np

class Trainer(object):
  def __init__(self, config, model_obj):
    self.config = config
    self.model = model_obj

  def get_train_op(self, loss, params, start_learning_rate, base_learning_rate, warmup_steps, scope=""):
    # add training op
    with tf.variable_scope(scope):
      self.global_step = tf.train.get_or_create_global_step()

      loss_l2 = tf.add_n([tf.nn.l2_loss(p) for p in params]) * self.config.l2_rate

      loss += loss_l2

      slope = (base_learning_rate - start_learning_rate) / warmup_steps
      warmup_rate = slope * tf.cast(self.global_step,
                                    tf.float32) + start_learning_rate

      decay_learning_rate = tf.train.exponential_decay(base_learning_rate, self.global_step,
                                                 1000, 0.96, staircase=True)
      learning_rate = tf.where(self.global_step < warmup_steps, warmup_rate,
                               decay_learning_rate)


      opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
      grads_and_vars = opt.compute_gradients(loss, params)
      gradients, variables = zip(*grads_and_vars)
      self.gradient_norm = tf.global_norm(gradients)
      clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5)
      self.param_norm = tf.global_norm(params)

      # Include batch norm mean and variance in gradient descent updates
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(update_ops):
        # Fetch self.updates to apply gradients to all trainable parameters.
        updates = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)

    return updates, learning_rate

  def train(self):
    update_op, scaffold, train_output_dic, dev_output_dic, test_output_dic = self.build_train_graph()
    with tf.train.MonitoredTrainingSession(checkpoint_dir=self.config.save_dir, scaffold=scaffold) as sess:
      tf.logging.info("start training:")
      for i in np.arange(self.config.training_iterations):
        sess.run(update_op)
        if (i % 100) == 0:
          tf.logging.info(i)
          tf.logging.info(sess.run(train_output_dic['loss']))


  def get_data_itaratoes(self):
    raise NotImplementedError()

  def build_train_graph(self):
    raise NotImplementedError()