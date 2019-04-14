import tensorflow as tf


class FeedforwardSelfAttention(object):
  def __init__(self, scope):
    self.scope = scope

  def create_vars(self, reuse=False):
    with tf.variable_scope(self.scope, reuse=reuse):
      # Initialize the weights and biases
      self.input_fully_connected_weights = tf.glorot_normal_initializer()
      self.input_fully_connected_biases = tf.zeros_initializer()


  def apply(self, input_keys, input_length, is_train=True):
    with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
      attention_score = tf.contrib.layers.fully_connected(input_keys,
                                                          num_outputs=1,
                                                          activation_fn=None,
                                                          weights_initializer=self.input_fully_connected_weights,
                                                          biases_initializer=self.input_fully_connected_biases)

      attention_mask = (1 - tf.cast(tf.sequence_mask(input_length), tf.float32)) * -999999999
      attention_score = attention_score + tf.expand_dims(attention_mask,-1)
      attention_score = tf.nn.softmax(attention_score)

      #attention_out = tf.squeeze(
      #  tf.matmul(tf.transpose(input_keys, perm=[0, 2, 1]), attention_score),axis=-1)

      attention_out = tf.multiply(input_keys, attention_score)
      tf.logging.info("attention output")
      tf.logging.info(attention_out)

    return attention_out


class MultiHeadScaledDotProductAttention(object):

  def __init__(self, hidden_dim, num_heads, attention_dropout_keepprob, scope="Attention"):
    if hidden_dim % num_heads != 0:
      raise ValueError("Hidden size must be evenly divisible by the number of "
                       "heads.")

    super(MultiHeadScaledDotProductAttention, self).__init__()
    self.hidden_dim = hidden_dim
    self.num_heads = num_heads
    self.attention_dropout_keepprob = attention_dropout_keepprob
    self.scope = scope

  def create_vars(self, reuse=tf.AUTO_REUSE):
    with tf.variable_scope(self.scope, reuse=reuse):
      # Layers for linearly projecting the queries, keys, and values.
      self.q_dense_layer = tf.layers.Dense(self.hidden_dim, use_bias=False, name="q")
      self.k_dense_layer = tf.layers.Dense(self.hidden_dim, use_bias=False, name="k")
      self.v_dense_layer = tf.layers.Dense(self.hidden_dim, use_bias=False, name="v")

      self.output_dense_layer = tf.layers.Dense(self.hidden_dim, use_bias=False,
                                                name="output_transform")

  def split_heads(self, x):
    """Split x into different heads, and transpose the resulting value.
    The tensor is transposed to insure the inner dimensions hold the correct
    values during the matrix multiplication.
    Args:
      x: A tensor with shape [batch_size, length, hidden_size]
    Returns:
      A tensor with shape [batch_size, num_heads, length, hidden_size/num_heads]
    """
    with tf.name_scope("split_heads"):
      batch_size = tf.shape(x)[0]
      length = tf.shape(x)[1]

      # Calculate depth of last dimension after it has been split.
      depth = (self.hidden_dim // self.num_heads)

      # Split the last dimension
      x = tf.reshape(x, [batch_size, length, self.num_heads, depth])

      # Transpose the result
      return tf.transpose(x, [0, 2, 1, 3])

  def combine_heads(self, x):
    """Combine tensor that has been split.
    Args:
      x: A tensor [batch_size, num_heads, length, hidden_size/num_heads]
    Returns:
      A tensor with shape [batch_size, length, hidden_size]
    """
    with tf.name_scope("combine_heads"):
      batch_size = tf.shape(x)[0]
      length = tf.shape(x)[2]
      x = tf.transpose(x, [0, 2, 1, 3])  # --> [batch, length, num_heads, depth]
      return tf.reshape(x, [batch_size, length, self.hidden_dim])

  def apply(self, x, y, bias, is_train=True, cache=None, x_presence=None, y_presence=None):
    """Apply attention mechanism to x and y.
    Args:
      x: a tensor with shape [batch_size, length_x, hidden_size]
      y: a tensor with shape [batch_size, length_y, hidden_size]
      bias: attention bias that will be added to the result of the dot product.
      cache: (Used during prediction) dictionary with tensors containing results
        of previous attentions. The dictionary must have the items:
            {"k": tensor with shape [batch_size, i, key_channels],
             "v": tensor with shape [batch_size, i, value_channels]}
        where i is the current decoded length.
    Returns:
      Attention layer output with shape [batch_size, length_x, hidden_size]
    """
    # Linearly project the query (q), key (k) and value (v) using different
    # learned projections. This is in preparation of splitting them into
    # multiple heads. Multi-head attention uses multiple queries, keys, and
    # values rather than regular attention (which uses a single q, k, v).
    tf.logging.info("attention x:")
    tf.logging.info(x)
    tf.logging.info("attention y:")
    tf.logging.info(y)
    tf.logging.info("attention bias:")
    tf.logging.info(bias)

    q = self.q_dense_layer(x)
    k = self.k_dense_layer(y)
    v = self.v_dense_layer(y)

    if cache is not None:
      # Combine cached keys and values with new keys and values.
      k = tf.concat([cache["k"], k], axis=1)
      v = tf.concat([cache["v"], v], axis=1)

      # Update cache
      cache["k"] = k
      cache["v"] = v

    # Split q, k, v into heads.
    q = self.split_heads(q)
    k = self.split_heads(k)
    v = self.split_heads(v)

    # Scale q to prevent the dot product between q and k from growing too large.
    depth = (self.hidden_dim // self.num_heads)
    q *= depth ** -0.5

    # Calculate dot product attention
    logits = tf.matmul(q, k, transpose_b=True)
    logits += bias
    weights = tf.nn.softmax(logits, name="attention_weights")
    if is_train:
      weights = tf.nn.dropout(weights, self.attention_dropout_keepprob)
    attention_output = tf.matmul(weights, v)

    # Recombine heads --> [batch_size, length, hidden_size]
    attention_output = self.combine_heads(attention_output)

    # Run the combined outputs through another linear projection layer.
    attention_output = self.output_dense_layer(attention_output)
    return attention_output


class ReversedMultiHeadScaledDotProductAttention(MultiHeadScaledDotProductAttention):

  def __init__(self, hidden_dim, num_heads, attention_dropout_keepprob, scope="ReversedAttention"):
    if hidden_dim % num_heads != 0:
      raise ValueError("Hidden size must be evenly divisible by the number of "
                       "heads.")

    super(ReversedMultiHeadScaledDotProductAttention, self).__init__(hidden_dim, num_heads,
                                                                     attention_dropout_keepprob, scope)

  def combine_heads(self, x):
    """Combine tensor that has been split.
    Args:
      x: A tensor [batch_size, num_heads, length, hidden_size/num_heads]
    Returns:
      A tensor with shape [batch_size, length, hidden_size]
    """
    with tf.name_scope("combine_heads"):
      batch_size = tf.shape(x)[0]
      length = tf.shape(x)[2]
      x = tf.transpose(x, [0, 2, 1, 3])  # --> [batch, length, num_heads, depth]
      return tf.reshape(x, [batch_size, length, self.hidden_dim])

  def apply(self, x, y, bias, x_presence=None, y_presence=None, presence_temp=1, is_train=True, cache=None):
    """Apply attention mechanism to x and y.
    Args:
      x: a tensor with shape [batch_size, length_x, hidden_size]
      y: a tensor with shape [batch_size, length_y, hidden_size]
      bias: attention bias that will be added to the result of the dot product.
      cache: (Used during prediction) dictionary with tensors containing results
        of previous attentions. The dictionary must have the items:
            {"k": tensor with shape [batch_size, i, key_channels],
             "v": tensor with shape [batch_size, i, value_channels]}
        where i is the current decoded length.
    Returns:
      Attention layer output with shape [batch_size, length_x, hidden_size]
    """
    # Linearly project the query (q), key (k) and value (v) using different
    # learned projections. This is in preparation of splitting them into
    # multiple heads. Multi-head attention uses multiple queries, keys, and
    # values rather than regular attention (which uses a single q, k, v).
    tf.logging.info("attention x:")
    tf.logging.info(x)
    tf.logging.info("attention y:")
    tf.logging.info(y)
    tf.logging.info("attention bias:")
    tf.logging.info(bias)

    q = self.q_dense_layer(x)
    k = self.k_dense_layer(y)
    v = self.v_dense_layer(y)

    if x_presence is None:
      x_presence = tf.ones((tf.shape(x)[0],tf.shape(x)[1],1))

    if y_presence is None:
      y_presence = tf.ones((tf.shape(y)[0],tf.shape(y)[1],1))

    if cache is not None:
      # Combine cached keys and values with new keys and values.
      k = tf.concat([cache["k"], k], axis=1)
      v = tf.concat([cache["v"], v], axis=1)

      # Update cache
      cache["k"] = k
      cache["v"] = v

    # Split q, k, v into heads.
    q = self.split_heads(q)
    k = self.split_heads(k)
    v = self.split_heads(v)

    # Scale q to prevent the dot product between q and k from growing too large.
    depth = (self.hidden_dim // self.num_heads)
    q *= depth ** -0.5

    # Calculate dot product attention
    # L_yD DL_x -> LyLx
    logits = tf.matmul(k, q, transpose_b=True)
    logits += bias
    # Pay less attention to the less present nodes.
    logits *= tf.tile(tf.expand_dims(x_presence,1), [1,tf.shape(logits)[1],1,1])

    # Normalize attention for keys(y nodes) for each head.
    assignment_probs = tf.nn.softmax(logits, axis=-1, name="attention_weights")
    assignment_weights = assignment_probs * tf.tile(tf.expand_dims(y_presence, axis=1), [1,tf.shape(logits)[1],1,1])

    # Aggregated attention of all heads:
    # [batch_size, num_heads, length y, length x] -> [batch_size, length y, length x]
    aggregated_assignment_weights = tf.reduce_sum(assignment_weights, axis=1)
    # [batch_size, length y, length x] -> [batch_size, length x]
    new_x_presence = tf.nn.softmax(tf.reduce_sum(aggregated_assignment_weights, axis=-2) ,axis=-1)

    if is_train:
      assignment_weights = tf.nn.dropout(assignment_weights, self.attention_dropout_keepprob)

    attention_output = tf.matmul(assignment_weights, v)


    # Recombine heads --> [batch_size, length, hidden_size]
    attention_output = self.combine_heads(attention_output)

    # Run the combined outputs through another linear projection layer.
    attention_output = self.output_dense_layer(attention_output)
    return attention_output, new_x_presence

