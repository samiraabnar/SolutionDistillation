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

      for i in np.arange(self.hparams.encoder_depth):
        self.dense_layers.append(tf.keras.layers.Dense(self.hparams.hidden_dim,
                                      activation=tf.nn.relu,
                                      use_bias=True))

      self.dense_layers.append(tf.keras.layers.Dense(self.hparams.output_dim,
                                    activation=tf.nn.relu,
                                    use_bias=True))


  def apply(self,examples, target_length=None, is_train=True, input_h=28, input_w=28, input_c=1, reuse=tf.AUTO_REUSE):
    inputs, targets, inputs_lengths, targets_lengths = examples

    with tf.variable_scope(self.scope, initializer=self.initializer, reuse=reuse):
      # Fully connected layers
      encoder_inputs = tf.cast(tf.reshape(inputs, [-1, input_h*input_w]), dtype=tf.float32)
      tf.logging.info("inputs")
      tf.logging.info(inputs)


      for i in np.arange(self.hparams.encoder_depth):
        encoder_inputs = self.dense_layers[i](encoder_inputs)


      # Output layer, 10 neurons for each digit
      logits = self.dense_layers[-1](encoder_inputs)[:,None,:]

      tf.logging.info("logits shape")
      tf.logging.info(logits)

      return {"logits": logits,
              "outputs": encoder_inputs,
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
