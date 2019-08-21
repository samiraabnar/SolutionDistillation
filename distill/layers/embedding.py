import tensorflow as tf

class Embedding(object):
  def __init__(self, pretrained_embedding_dim, tuned_embedding_dim, keep_prob, vocab_size=None, scope="EmbeddingLayer"):
    if vocab_size is not None:
      self.vocab_size = vocab_size
    else:
      self.vocab_size = pretrained_embedding_dim.shape[0]

    self.pretrained_embedding_dim = pretrained_embedding_dim
    self.keep_prob = keep_prob
    self.scope = scope
    self.tuned_embedding_dim = tuned_embedding_dim


  def create_vars(self, pretrained_word_embeddings=None, reuse=False):
    # Create the embeddings
    with tf.variable_scope(self.scope, reuse=reuse):
      with tf.variable_scope("Embeddings"):
        if self.pretrained_embedding_dim > 0:
          self.fixed_embedding = tf.get_variable("fixed_word_emb_matrix",
                                                 dtype=tf.float32,
                                                 initializer=pretrained_word_embeddings,
                                                 trainable=False)
        if self.tuned_embedding_dim > 0:
          self.tuned_embedding = tf.get_variable("tuned_word_emb_matrix",
                                                 shape=(self.vocab_size, self.tuned_embedding_dim),
                                                 dtype=tf.float32,
                                                 initializer=tf.truncated_normal_initializer,
                                                 trainable=True)

        self.default_embedding = tf.get_variable("default_word_emb_matrix",
                                               shape=(self.vocab_size, 10),
                                               dtype=tf.float32,
                                               initializer=tf.truncated_normal_initializer,
                                               trainable=True)


  def apply(self, inputs, reuse=tf.AUTO_REUSE, is_train=True):
    with tf.variable_scope(self.scope, reuse=reuse):
      fixed_embedded_input = None
      embedded_input = tf.nn.embedding_lookup(self.default_embedding, inputs)

      if self.pretrained_embedding_dim > 0:
          fixed_embedded_input = tf.nn.embedding_lookup(self.fixed_embedding, inputs)
      if self.tuned_embedding_dim > 0:
        tuned_embedded_input = tf.nn.embedding_lookup(self.tuned_embedding, inputs)
        if fixed_embedded_input is not None:
          embedded_input = tf.concat([fixed_embedded_input, tuned_embedded_input], axis=-1)

      if is_train:
        embedded_input = tf.nn.dropout(embedded_input, self.keep_prob)

    return embedded_input


class EmbeddingSharedWeights(object):
  """Calculates input embeddings and pre-softmax linear with shared weights."""

  def __init__(self, vocab_size, embedding_dim, pretrained_embeddings=None, method="gather", scope='SharedEmbedding'):
    """Specify characteristic parameters of embedding layer.
    Args:
      vocab_size: Number of tokens in the embedding. (Typically ~32,000)
      hidden_size: Dimensionality of the embedding. (Typically 512 or 1024)
      method: Strategy for performing embedding lookup. "gather" uses tf.gather
        which performs well on CPUs and GPUs, but very poorly on TPUs. "matmul"
        one-hot encodes the indicies and formulates the embedding as a sparse
        matrix multiplication. The matmul formulation is wasteful as it does
        extra work, however matrix multiplication is very fast on TPUs which
        makes "matmul" considerably faster than "gather" on TPUs.
    """
    super(EmbeddingSharedWeights, self).__init__()
    self.vocab_size = vocab_size
    self.hidden_size = embedding_dim
    if method not in ("gather", "matmul"):
      raise ValueError("method {} must be 'gather' or 'matmul'".format(method))
    self.method = method
    self.scope = scope
    self.pretrained_embeddings = pretrained_embeddings
    if self.pretrained_embeddings is None:
      self.initializer = tf.contrib.layers.xavier_initializer()
      # tf.random_normal_initializer(
      #           0., self.hidden_size ** -0.5)
    else:
      self.initializer = pretrained_embeddings

  def create_vars(self, is_train=True):
    with tf.variable_scope(self.scope):
      with tf.variable_scope("embedding_and_softmax", reuse=tf.AUTO_REUSE):
        # Create and initialize weights. The random normal initializer was chosen
        # randomly, and works well.
        self.shared_weights = tf.get_variable(
            "weights", shape=[self.vocab_size, self.hidden_size] if self.pretrained_embeddings is None else None,
            initializer=self.initializer,
            trainable=is_train)

      self.built = True

  def apply(self, x):
    """Get token embeddings of x.
    Args:
      x: An int64 tensor with shape [batch_size, length]
    Returns:
      embeddings: float32 tensor with shape [batch_size, length, embedding_size]
      padding: float32 tensor with shape [batch_size, length] indicating the
        locations of the padding tokens in x.
    """
    with tf.variable_scope(self.scope, reuse=True):
      with tf.name_scope("embedding"):
        # Create binary mask of size [batch_size, length]
        mask = tf.to_float(tf.not_equal(x, 0))

        embeddings = tf.gather(self.shared_weights, x)
        embeddings *= tf.expand_dims(mask, -1)


      # Scale embedding by the sqrt of the hidden size
      embeddings *= self.hidden_size ** 0.5

      return embeddings

  def linear(self, x):
    """Computes logits by running x through a linear layer.
    Args:
      x: A float32 tensor with shape [batch_size, length, hidden_size]
    Returns:
      float32 tensor with shape [batch_size, length, vocab_size].
    """
    with tf.variable_scope(self.scope, reuse=True):
      with tf.name_scope("presoftmax_linear"):
        batch_size = tf.shape(x)[0]
        length = tf.shape(x)[1]

        x = tf.reshape(x, [-1, self.hidden_size])
        logits = tf.matmul(x, self.shared_weights, transpose_b=True)

      return tf.reshape(logits, [batch_size, length, self.vocab_size])
