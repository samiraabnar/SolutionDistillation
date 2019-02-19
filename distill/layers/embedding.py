import tensorflow as tf

class Embedding(object):
  def __init__(self, vocab_size=None, embedding_dim=None, keep_prob=1.0, scope="EmbeddingLayer"):
    self.vocab_size = vocab_size
    self.embedding_dim = embedding_dim
    self.keep_prob = keep_prob
    self.scope = scope


  def create_vars(self, pretrained_word_embeddings, reuse=False):
    # Create the embeddings
    with tf.variable_scope(self.scope, reuse=reuse):
      with tf.variable_scope("Embeddings"):
        self.embedding = tf.get_variable("word_emb_matrix",
                                               dtype=tf.float32,
                                               initializer=pretrained_word_embeddings,
                                               trainable=False)


  def apply(self, inputs, reuse=tf.AUTO_REUSE):
    with tf.variable_scope(self.scope, reuse=reuse):
      embedded_input = tf.nn.embedding_lookup(self.embedding, inputs)
      embedded_input = tf.nn.dropout(embedded_input, self.keep_prob)

    return embedded_input

