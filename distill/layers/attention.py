import tensorflow as tf


class FeedforwardSelfAttention(object):
  def __init__(self, scope):
    self.scope = scope

  def create_vars(self, reuse=False):
    with tf.variable_scope(self.scope, reuse=reuse):
      # Initialize the weights and biases
      self.input_fully_connected_weights = tf.truncated_normal_initializer(stddev=0.1)
      self.input_fully_connected_biases = tf.zeros_initializer()


  def apply(self, input_keys):
    with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
      attention_score = tf.contrib.layers.fully_connected(input_keys,
                                                          num_outputs=1,
                                                          weights_initializer=self.input_fully_connected_weights,
                                                          biases_initializer=self.input_fully_connected_biases)
      attention_score = tf.nn.softmax(attention_score)
      #attention_out = tf.squeeze(
      #  tf.matmul(tf.transpose(input_keys, perm=[0, 2, 1]), attention_score),axis=-1)

      attention_out = tf.math.multiply(input_keys, attention_score)
      tf.logging.info("attention output")
      tf.logging.info(attention_out)

    return attention_out

