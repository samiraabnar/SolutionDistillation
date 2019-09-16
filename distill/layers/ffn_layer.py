import tensorflow as tf

class DenseLayer(object):
  def __init__(self, hidden_dim, scope="dense"):
    self.hidden_dim = hidden_dim
    self.scope = scope

  def create_vars(self, reuse=False):
    with tf.variable_scope(self.scope, reuse=reuse):
      self.layer  = tf.keras.layers.Dense(self.hidden_dim,
                                        activation=tf.nn.relu,
                                        use_bias=True)

  def apply(self, x, is_train=True, reuse=tf.AUTO_REUSE):
    with tf.variable_scope(self.scope, reuse=reuse):
      return self.layer(x)

class FeedFowardNetwork(object):
  """Fully connected feedforward network."""

  def __init__(self, hidden_size, filter_size, relu_dropout_keepprob, allow_pad, scope="FF"):
    self.hidden_size = hidden_size
    self.filter_size = filter_size
    self.relu_dropout_keepprob = relu_dropout_keepprob
    self.allow_pad = allow_pad
    self.scope =scope



  def create_vars(self, reuse=False):
    with tf.variable_scope(self.scope, reuse=reuse):
      self.filter_dense_layer = tf.layers.Dense(
      self.filter_size, use_bias=True, activation=tf.nn.relu, name="filter_layer")
      self.output_dense_layer = tf.layers.Dense(
          self.hidden_size, use_bias=True, name="output_layer")

  def apply(self, x, is_train=True, padding=None, reuse=False):
    """Return outputs of the feedforward network.
    Args:
      x: tensor with shape [batch_size, length, hidden_size]
      padding: (optional) If set, the padding values are temporarily removed
        from x (provided self.allow_pad is set). The padding values are placed
        back in the output tensor in the same locations.
        shape [batch_size, length]
    Returns:
      Output of the feedforward network.
      tensor with shape [batch_size, length, hidden_size]
    """
    with tf.variable_scope(self.scope, reuse=reuse):
      padding = None if not self.allow_pad else padding

      # Retrieve dynamically known shapes
      batch_size = tf.shape(x)[0]
      length = tf.shape(x)[1]

      if padding is not None:
        with tf.name_scope("remove_padding"):
          # Flatten padding to [batch_size*length]
          pad_mask = tf.reshape(padding, [-1])

          nonpad_ids = tf.to_int32(tf.where(pad_mask < 1e-9))

          # Reshape x to [batch_size*length, hidden_size] to remove padding
          x = tf.reshape(x, [-1, self.hidden_size])
          x = tf.gather_nd(x, indices=nonpad_ids)

          # Reshape x from 2 dimensions to 3 dimensions.
          x.set_shape([None, self.hidden_size])
          x = tf.expand_dims(x, axis=0)

      output = self.filter_dense_layer(x)
      if is_train:
        output = tf.nn.dropout(output, self.relu_dropout_keepprob)
      output = self.output_dense_layer(output)

      if padding is not None:
        with tf.name_scope("re_add_padding"):
          output = tf.squeeze(output, axis=0)
          output = tf.scatter_nd(
              indices=nonpad_ids,
              updates=output,
              shape=[batch_size * length, self.hidden_size]
          )
          output = tf.reshape(output, [batch_size, length, self.hidden_size])
    return output