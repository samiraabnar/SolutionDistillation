import tensorflow as tf
import numpy as np

class Trainer(object):
  def __init__(self, config, model_obj):
    self.config = config
    self.model = model_obj

  def get_train_op(self, loss, params, start_learning_rate, base_learning_rate, warmup_steps,
                   clip_gradient_norm=0,  scope=""):
    # add training op
    with tf.variable_scope(scope):
      self.global_step = tf.train.get_or_create_global_step()

      loss_l2 = tf.add_n([tf.nn.l2_loss(p) for p in params]) * self.config.l2_rate

      loss += loss_l2

      slope = (base_learning_rate - start_learning_rate) / warmup_steps
      warmup_rate = slope * tf.cast(self.global_step,
                                    tf.float32) + start_learning_rate

      if self.config.decay_learning_rate:
        decay_learning_rate = tf.train.exponential_decay(base_learning_rate, self.global_step,
                                                       100, 0.98, staircase=False)
      else:
        decay_learning_rate = base_learning_rate

      learning_rate = tf.where(self.global_step < warmup_steps, warmup_rate,
                               decay_learning_rate)


      opt = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                   beta1=self.model.hparams.optimizer_adam_beta1,
                                   beta2=self.model.hparams.optimizer_adam_beta2,
                                   epsilon=self.model.hparams.optimizer_adam_epsilon)
      grads_and_vars = opt.compute_gradients(loss, params)
      gradients, variables = zip(*grads_and_vars)
      if clip_gradient_norm > 0:
        gradient_norm, _ = tf.clip_by_global_norm(gradients, clip_gradient_norm)
      self.param_norm = tf.global_norm(params)

      # Include batch norm mean and variance in gradient descent updates
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      updates = opt.apply_gradients(zip(gradients, params), global_step=self.global_step)

      # Create an ExponentialMovingAverage object
      ema = tf.train.ExponentialMovingAverage(decay=0.9999)

      with tf.control_dependencies([updates]+update_ops):
        training_op = ema.apply(tf.trainable_variables())

    return training_op, learning_rate


  def train(self):
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
      update_op, scaffold, train_output_dic, _, _ = self.build_train_graph()


      with tf.train.MonitoredTrainingSession(checkpoint_dir=self.config.save_dir, scaffold=scaffold) as sess:
        tf.logging.info("start training:")
        tf.logging.info(self.config.training_iterations)
        for i in np.arange(self.config.training_iterations):
          sess.run(update_op)



  def get_data_itaratoes(self):
    raise NotImplementedError()

  def build_train_graph(self):
    raise NotImplementedError()

  def build_eval_graph(self):
    raise NotImplementedError()