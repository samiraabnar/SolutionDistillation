import tensorflow as tf


class LayerNormalization(object):
  """Applies layer normalization."""

  def __init__(self, hidden_size):
    self.hidden_size = hidden_size

  def create_vars(self):
    self.scale = tf.get_variable("layer_norm_scale", [self.hidden_size],
                                 initializer=tf.ones_initializer())
    self.bias = tf.get_variable("layer_norm_bias", [self.hidden_size],
                                initializer=tf.zeros_initializer())
    self.built = True

  def apply(self, x, epsilon=1e-6):
    mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
    variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
    norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
    return norm_x * self.scale + self.bias


class PrePostProcessingWrapper(object):
  """Wrapper class that applies layer pre-processing and post-processing."""

  def __init__(self, layer, hidden_dim, postprocess_dropout_keepprob):
    self.layer = layer
    self.hidden_dim = hidden_dim
    self.postprocess_dropout_keepprob = postprocess_dropout_keepprob



  def create_vars(self, **kwargs):
    # Create normalization layer
    self.layer_norm = LayerNormalization(self.hidden_dim)
    self.layer_norm.create_vars()
    self.layer.create_vars(**kwargs)

  def apply(self, x, is_train=True, **kwargs):
    # Preprocessing: apply layer normalization
    y = self.layer_norm.apply(x)

    # Get layer output
    y = self.layer.apply(y, is_train=is_train, **kwargs)

    extra = None
    if isinstance(y, tuple):
      y, extra = y

    # Postprocessing: apply dropout and residual connection
    if is_train:
      y = tf.nn.dropout(y, self.postprocess_dropout_keepprob)

    return x + y , extra