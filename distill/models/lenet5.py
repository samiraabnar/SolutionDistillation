import tensorflow as tf
import numpy as np
from distill.data_util.prep_mnist import Mnist1D
from distill.layers.cnn import ConvLayer


class LeNet5(object):
  def __init__(self, hparams, task, scope="lenet5"):
    self.hparams = hparams
    self.filter_size = hparams.filter_size
    self.scope = scope
    self.task = task
    self.initializer = tf.variance_scaling_initializer(
      self.hparams.initializer_gain, mode="fan_avg", distribution="uniform")


  def create_vars(self, reuse=False):
    with tf.variable_scope(self.scope, initializer=self.initializer, reuse=reuse):
      self.cnn_layers = []
      self.dense_layers = []
      convpool1 = ConvLayer(filter_h=5, filter_w=5, pool_size=2,
                            in_channels=1, out_channels=6,
                            scope="convpool1")
      convpool1.create_vars(reuse=reuse)
      convpool2 = ConvLayer(filter_h=5, filter_w=5, pool_size=2,
                            in_channels=6, out_channels=16,
                            scope="convpool1")
      convpool2.create_vars(reuse=reuse)

      dens1 = tf.keras.layers.Dense(120,
                                    activation=tf.nn.relu,
                                    use_bias=True)
      dens2 = tf.keras.layers.Dense(84,
                                    activation=tf.nn.relu,
                                    use_bias=True)

      dens3 = tf.keras.layers.Dense(10,
                                    activation=tf.nn.relu,
                                    use_bias=True)

      self.cnn_layers = [convpool1, convpool2]
      self.dense_layers = [dens1, dens2, dens3]

  def apply(self,examples, target_length=None, is_train=True, input_h=28, input_w=28, input_c=1, reuse=tf.AUTO_REUSE):
    inputs, targets, inputs_lengths, targets_lengths = examples

    with tf.variable_scope(self.scope, initializer=self.initializer, reuse=reuse):
      # Here we defind the CNN architecture (LeNet-5)

      # Reshape input to 4-D vector
      input_layer = tf.cast(tf.reshape(inputs, [-1, input_h, input_w, input_c]), dtype=tf.float32)  # -1 adds minibatch support.
      # Padding the input to make it 32x32. Specification of LeNET
      padded_input = tf.pad(input_layer, [[0, 0], [2, 2], [2, 2], [0, 0]], "CONSTANT")

      # Convolutional Layer #1
      # Has a default stride of 1
      # Output: 28 * 28 * 6
      # Pooling Layer #1
      # Sampling half the output of previous layer
      # Output: 14 * 14 * 6
      convpool1_output = self.cnn_layers[0].apply(padded_input)
      tf.logging.info(convpool1_output)

      # Convolutional Layer #2
      # Output: 10 * 10 * 16
      # Pooling Layer #2
      # Output: 5 * 5 * 16
      convpool2_output = self.cnn_layers[1].apply(convpool1_output)
      tf.logging.info(convpool2_output)


      # Reshaping output into a single dimention array for input to fully connected layer
      pool2_flat = tf.reshape(convpool2_output, [-1, 5 * 5 * 16])

      # Fully connected layer #1: Has 120 neurons
      dense1 = self.dense_layers[0](pool2_flat)

      # Fully connected layer #2: Has 84 neurons
      dense2 =  self.dense_layers[1](dense1)

      # Output layer, 10 neurons for each digit
      logits = self.dense_layers[2](dense2)

      return {"logits": logits,
              "outputs": dense2,
              "inputs": inputs,
              "targets": targets
            }



# if __name__ == '__main__':
#
#   class Config(object):
#     def __init__(self):
#       self.output_dim = 10
#       self.input_dim = 728
#       self.encoder_depth = 1
#       self.decoder_depth = 1
#       self.sent_rep_mode = "all"
#       self.scope = "lenet5"
#       self.batch_size = 64
#       self.input_dropout_keep_prob = 1.0
#       self.hidden_dropout_keep_prob = 1.0
#       self.number_of_heads = 2
#       self.filter_size = 5
#       self.initializer_gain = 1.0
#       self.label_smoothing = 0.1
#       self.clip_grad_norm = 0.  # i.e. no gradient clipping
#       self.optimizer_adam_epsilon = 1e-9
#       self.learning_rate = 0.001
#       self.learning_rate_warmup_steps = 1000
#       self.initializer_gain = 1.0
#       self.initializer = "uniform_unit_scaling"
#       self.weight_decay = 0.0
#       self.optimizer_adam_beta1 = 0.9
#       self.optimizer_adam_beta2 = 0.98
#       self.num_sampled_classes = 0
#       self.label_smoothing = 0.1
#       self.clip_grad_norm = 0.  # i.e. no gradient clipping
#       self.optimizer_adam_epsilon = 1e-9
#       self.alpha = 1
#
#
#   tf.logging.set_verbosity(tf.logging.INFO)
#   bin_iden = Mnist1D('data/mnist1d')
#
#
#   dataset = tf.data.TFRecordDataset(bin_iden.get_tfrecord_path(mode="train"))
#   dataset = dataset.map(bin_iden.parse_examples)
#   dataset = dataset.padded_batch(5, padded_shapes=bin_iden.get_padded_shapes())
#   iterator = dataset.make_initializable_iterator()
#
#   example = iterator.get_next()
#   inputs, targets, inputs_lengths, targets_length = example
#
#   lenet = LeNet5(Config(),
#                        task=bin_iden,
#                        scope="lenet5")
#   lenet.create_vars(reuse=False)
#
#   outputs = lenet.apply(example, target_length=bin_iden.target_length, is_train=True)
#   outputs = outputs["logits"]
#
#   global_step = tf.train.get_or_create_global_step()
#   scaffold = tf.train.Scaffold(local_init_op=tf.group(tf.local_variables_initializer(),
#                                                       iterator.initializer))
#   with tf.train.MonitoredTrainingSession(checkpoint_dir='logs/test_lenet', scaffold=scaffold) as sess:
#     for _ in np.arange(1):
#       inp, targ, outp = sess.run([inputs, targets, outputs])
#
#       print(inp.shape)
#       print(targ.shape)
#       print(outp.shape)
