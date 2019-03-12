import tensorflow as tf

class Embedding(object):
  def __init__(self, vocab_size=None, embedding_dim=None, tuned_embedding_dim=5, keep_prob=1.0, scope="EmbeddingLayer"):
    self.vocab_size = vocab_size
    self.embedding_dim = embedding_dim
    self.keep_prob = keep_prob
    self.scope = scope
    self.tuned_embedding_dim = tuned_embedding_dim


  def create_vars(self, pretrained_word_embeddings, reuse=False):
    # Create the embeddings
    with tf.variable_scope(self.scope, reuse=reuse):
      with tf.variable_scope("Embeddings"):
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


  def apply(self, inputs, reuse=tf.AUTO_REUSE, is_train=True):
    with tf.variable_scope(self.scope, reuse=reuse):
      fixed_embedded_input = tf.nn.embedding_lookup(self.fixed_embedding, inputs)
      if self.tuned_embedding_dim > 0:
        tuned_embedded_input = tf.nn.embedding_lookup(self.tuned_embedding, inputs)
        embedded_input = tf.concat([fixed_embedded_input, tuned_embedded_input], axis=-1)
      else:
        embedded_input = fixed_embedded_input
      if is_train:
        embedded_input = tf.nn.dropout(embedded_input, self.keep_prob)

    return embedded_input

