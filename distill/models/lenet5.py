import tensorflow as tf

class LeNet5(object):
  def __init__(self, hparams, task, scope="lenet5"):
    self.hparams = hparams
    self.vocab_size = hparams.vocab_size
    self.hidden_dim = hparams.hidden_dim
    self.number_of_heads = hparams.number_of_heads
    self.encoder_depth = hparams.encoder_depth
    self.ff_filter_size = hparams.ff_filter_size
    self.attention_dropout_keepprob = hparams.attention_dropout_keepprob
    self.relu_dropout_keepprob = hparams.relu_dropout_keepprob
    self.postprocess_dropout_keepprob = hparams.postprocess_dropout_keepprob
    self.initializer_gain = hparams.initializer_gain
    self.scope = scope
    self.task = task
    self.initializer = tf.variance_scaling_initializer(
      self.initializer_gain, mode="fan_avg", distribution="uniform")


  def create_vars(self, reuse=False):
    with tf.variable_scope(self.scope, initializer=self.initializer, reuse=reuse):

  def apply(self,examples, reuse=tf.AUTO_REUSE):
    inputs, targets, inputs_lengths, targets_lengths = examples

    with tf.variable_scope(self.scope, initializer=self.initializer, reuse=reuse):
    # Here we defind the CNN architecture (LeNet-5)

    # Reshape input to 4-D vector
    input_layer = tf.reshape(inputs, [-1, 28, 28, 1])  # -1 adds minibatch support.

    # Padding the input to make it 32x32. Specification of LeNET
    padded_input = tf.pad(input_layer, [[0, 0], [2, 2], [2, 2], [0, 0]], "CONSTANT")

    # Convolutional Layer #1
    # Has a default stride of 1
    # Output: 28 * 28 * 6
    conv1 = tf.layers.conv2d(
      inputs=padded_input,
      filters=6,  # Number of filters.
      kernel_size=5,  # Size of each filter is 5x5.
      padding="valid",  # No padding is applied to the input.
      activation=tf.nn.relu)

    # Pooling Layer #1
    # Sampling half the output of previous layer
    # Output: 14 * 14 * 6
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2
    # Output: 10 * 10 * 16
    conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=16,  # Number of filters
      kernel_size=5,  # Size of each filter is 5x5
      padding="valid",  # No padding
      activation=tf.nn.relu)

    # Pooling Layer #2
    # Output: 5 * 5 * 16
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Reshaping output into a single dimention array for input to fully connected layer
    pool2_flat = tf.reshape(pool2, [-1, 5 * 5 * 16])

    # Fully connected layer #1: Has 120 neurons
    dense1 = tf.layers.dense(inputs=pool2_flat, units=120, activation=tf.nn.relu)

    # Fully connected layer #2: Has 84 neurons
    dense2 = tf.layers.dense(inputs=dense1, units=84, activation=tf.nn.relu)

    # Output layer, 10 neurons for each digit
    logits = tf.layers.dense(inputs=dense2, units=10)

    return logits

  # Pass the input thorough our CNN
  logits = CNN(X)
  softmax = tf.nn.softmax(logits)

  # Convert our labels into one-hot-vectors
  labels = tf.one_hot(indices=tf.cast(Y, tf.int32), depth=10)

  # Compute the cross-entropy loss
  cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,
                                                                   labels=labels))

  # Use adam optimizer to reduce cost
  optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
  train_op = optimizer.minimize(cost)

  # For testing and prediction
  predictions = tf.argmax(softmax, axis=1)


