import tensorflow as tf
import numpy as np
from distill.data_util.prep_mnist import Mnist1D


class VanillaFF(object):
  def __init__(self, hparams, task, scope="vanilla_ff"):
    self.hparams = hparams
    self.scope = scope
    self.task = task
    self.initializer = tf.variance_scaling_initializer(
      self.hparams.initializer_gain, mode="fan_avg", distribution="uniform")


  def create_vars(self, reuse=False, pretrained_embeddings=None):
    with tf.variable_scope(self.scope, initializer=self.initializer, reuse=reuse):
      self.dense_layers = []

      dens1 = tf.keras.layers.Dense(728,
                                    activation=tf.nn.relu,
                                    use_bias=True)
      dens2 = tf.keras.layers.Dense(256,
                                    activation=tf.nn.relu,
                                    use_bias=True)
      dens3 = tf.keras.layers.Dense(120,
                                    activation=tf.nn.relu,
                                    use_bias=True)
      dens4 = tf.keras.layers.Dense(84,
                                    activation=tf.nn.relu,
                                    use_bias=True)

      dens5 = tf.keras.layers.Dense(10,
                                    activation=tf.nn.relu,
                                    use_bias=True)

      self.dense_layers = [dens1, dens2, dens3, dens4, dens5]

  def apply(self,examples, target_length=None, is_train=True, input_h=28, input_w=28, input_c=1, reuse=tf.AUTO_REUSE):
    inputs, targets, inputs_lengths, targets_lengths = examples

    with tf.variable_scope(self.scope, initializer=self.initializer, reuse=reuse):
      # Fully connected layers
      inputs = tf.cast(tf.reshape(inputs, [-1, input_h*input_w]), dtype=tf.float32)
      tf.logging.info("inputs")
      tf.logging.info(inputs)

      dense1 = self.dense_layers[0](inputs)
      dense2 = self.dense_layers[1](dense1)
      dense3 = self.dense_layers[2](dense2)
      dense4 = self.dense_layers[3](dense3)

      # Output layer, 10 neurons for each digit
      logits = self.dense_layers[4](dense4)[:,None,:]

      tf.logging.info("logits shape")
      tf.logging.info(logits)

      return {"logits": logits,
              "outputs": dense4,
              "inputs": inputs,
              "targets": targets,
              "trainable_vars": tf.trainable_variables(scope=self.scope),
            }



if __name__ == '__main__':

  class Config(object):
    def __init__(self):
      self.output_dim = 10
      self.input_dim = 728
      self.encoder_depth = 1
      self.decoder_depth = 1
      self.sent_rep_mode = "all"
      self.scope = "lenet5"
      self.batch_size = 64
      self.input_dropout_keep_prob = 1.0
      self.hidden_dropout_keep_prob = 1.0
      self.number_of_heads = 2
      self.filter_size = 5
      self.initializer_gain = 1.0
      self.label_smoothing = 0.1
      self.clip_grad_norm = 0.  # i.e. no gradient clipping
      self.optimizer_adam_epsilon = 1e-9
      self.learning_rate = 0.001
      self.learning_rate_warmup_steps = 1000
      self.initializer_gain = 1.0
      self.initializer = "uniform_unit_scaling"
      self.weight_decay = 0.0
      self.optimizer_adam_beta1 = 0.9
      self.optimizer_adam_beta2 = 0.98
      self.num_sampled_classes = 0
      self.label_smoothing = 0.1
      self.clip_grad_norm = 0.  # i.e. no gradient clipping
      self.optimizer_adam_epsilon = 1e-9
      self.alpha = 1


  tf.logging.set_verbosity(tf.logging.INFO)
  bin_iden = Mnist1D('data/mnist1d')


  dataset = tf.data.TFRecordDataset(bin_iden.get_tfrecord_path(mode="train"))
  dataset = dataset.map(bin_iden.parse_examples)
  dataset = dataset.padded_batch(5, padded_shapes=bin_iden.get_padded_shapes())
  iterator = dataset.make_initializable_iterator()

  example = iterator.get_next()
  inputs, targets, inputs_lengths, targets_length = example

  lenet = VanillaFF(Config(),
                       task=bin_iden,
                       scope="vanillaff")
  lenet.create_vars(reuse=False)

  outputs = lenet.apply(example, target_length=bin_iden.target_length, is_train=True)
  outputs = outputs["logits"]

  global_step = tf.train.get_or_create_global_step()
  scaffold = tf.train.Scaffold(local_init_op=tf.group(tf.local_variables_initializer(),
                                                      iterator.initializer))
  with tf.train.MonitoredTrainingSession(checkpoint_dir='logs/test_vanillaff', scaffold=scaffold) as sess:
    for _ in np.arange(1):
      inp, targ, outp = sess.run([inputs, targets, outputs])

      print(inp.shape)
      print(targ.shape)
      print(outp.shape)
