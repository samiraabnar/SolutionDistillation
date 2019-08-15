import tensorflow as tf

from distill.layers.embedding import Embedding
from distill.layers.lstm import LSTM
from distill.layers.bilstm import BiLSTM



class LmLSTM(object):
  def __init__(self, config, task, model=LSTM, scope="LMLSTM"):
    self.config = config
    self.scope=scope
    self.task = task
    self.eos_id = self.task.eos_id
    self.lstm = model(hidden_dim=config.hidden_dim,
                      output_dim=config.vocab_size,
                      hidden_keep_prob=config.input_dropout_keep_prob,
                      attention_mechanism=self.config.attention_mechanism,
                      depth=config.depth,
                      sent_rep_mode=self.config.sent_rep_mode,
                      scope=scope)


  def build_graph(self):
    with tf.variable_scope(self.scope):
      self.embedding_layer = Embedding(vocab_size=self.config.vocab_size,
                                       tuned_embedding_dim=self.config.embedding_dim,
                                       keep_prob=self.config.input_dropout_keep_prob)
      self.embedding_layer.create_vars()

      self.lstm.create_vars()

      # Output embedding
      self.output_embedding_mat = tf.get_variable("output_embedding_mat",
                                                  [self.config.vocab_size, self.config.hidden_dim],
                                                  dtype=tf.float32)

      self.output_embedding_bias = tf.get_variable("output_embedding_bias",
                                                   [self.config.vocab_size],
                                                   dtype=tf.float32)


  def apply(self, examples, is_train=True, reuse=tf.AUTO_REUSE):
    inputs, labels, inputs_length = examples
    tf.logging.info(inputs_length)
    inputs_mask = tf.sequence_mask(inputs_length)

    batch_size = tf.shape(inputs)[0]
    with tf.variable_scope(self.scope, reuse=reuse):
      embedded_input = self.embedding_layer.apply(inputs, is_train)
      lstm_output_dic = self.lstm.apply(inputs=embedded_input, inputs_length=inputs_length, is_train=is_train)

      seq_states = lstm_output_dic['raw_outputs']

      def output_embedding(current_output):
        return tf.add(
          tf.matmul(current_output, tf.transpose(self.output_embedding_mat)),
          self.output_embedding_bias)

      logits = tf.map_fn(output_embedding, seq_states)

      tf.logging.info("states")
      tf.logging.info(seq_states)
      tf.logging.info("logits")
      tf.logging.info(logits)
      predictions = tf.argmax(logits, axis=-1)

      flat_logits = tf.reshape(logits, [-1, tf.shape(logits)[-1]])
      tf.logging.info(flat_logits)

      flat_labels = tf.reshape(labels, [-1])
      tf.logging.info(flat_labels)

      flat_mask = tf.cast(tf.reshape(inputs_mask, [-1]), tf.float32)

      flat_predictions = tf.reshape(predictions, [-1])

      if is_train:
        loss = tf.nn.sampled_softmax_loss(
          weights=self.output_embedding_mat,
          biases=self.output_embedding_bias,
          labels=tf.reshape(labels, [-1, 1]),
          inputs=tf.reshape(seq_states, [-1, 128]),
          num_classes=self.config.vocab_size,
          num_sampled=1000,
          partition_strategy="div")
      else:
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=flat_labels,
          logits=flat_logits) * flat_mask

      perplexity = tf.reduce_mean(tf.exp(loss))
      loss = tf.reduce_mean(loss)

      tf.logging.info(flat_labels)
      tf.logging.info(flat_mask)
      accuracy = tf.reduce_sum(tf.cast(tf.equal(flat_labels, flat_predictions), dtype=tf.float32) * flat_mask) / tf.reduce_sum(flat_mask)
      correct_predictions = tf.cast(tf.equal(predictions, labels), dtype=tf.float32) * tf.cast(inputs_mask, dtype=tf.float32)
      correct_sequences = tf.reduce_min(correct_predictions, axis=-1)
      sequence_accuracy = tf.reduce_sum(correct_sequences) / tf.cast(batch_size, dtype=tf.float32)

    return {'inputs': inputs,
            'loss': loss,
            'predictions': predictions,
            'logits': logits,
            'accuracy': accuracy,
            'sequence_accuracy': sequence_accuracy,
            'perplexity': perplexity,
            'trainable_vars': tf.trainable_variables(scope=self.scope)}
