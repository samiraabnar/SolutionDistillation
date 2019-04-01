import tensorflow as tf

from distill.layers.embedding import Embedding
from distill.layers.lstm import LSTM
from distill.layers.bilstm import BiLSTM



class SentimentLSTM(object):
  def __init__(self, hparams, model=LSTM, scope="SentimentLSTM"):
    self.hparams = hparams
    self.scope=scope
    self.lstm = model(hidden_dim=hparams.hidden_dim,
                      output_dim=hparams.output_dim,
                      hidden_keep_prob=hparams.input_dropout_keep_prob,
                      attention_mechanism=self.hparams.attention_mechanism,
                      depth=hparams.depth,
                      sent_rep_mode=self.hparams.sent_rep_mode,
                      scope=scope)


  def build_graph(self, pretrained_word_embeddings):
    with tf.variable_scope(self.scope):
      self.embedding_layer = Embedding(vocab_size=self.hparams.vocab_size,
                                       pretrained_embedding_dim=self.hparams.embedding_dim,
                                       keep_prob=self.hparams.input_dropout_keep_prob,
                                       tuned_embedding_dim=0)
      self.embedding_layer.create_vars(pretrained_word_embeddings=pretrained_word_embeddings)

      self.lstm.create_vars()

      # Create the fully connected layers
      with tf.variable_scope("Projection"):
        # Initialize the weights and biases
        self.output_fully_connected_weights = tf.contrib.layers.xavier_initializer()


  def apply(self, examples, is_train=True, reuse=tf.AUTO_REUSE):
    example_id, length, is_leaf, left_children, right_children, node_word_ids, labels, binary_labels, \
    root_label, root_binary_label, seq_lengths, seq_inputs = examples

    labels = root_binary_label

    batch_size = tf.shape(seq_inputs)[0]
    with tf.variable_scope(self.scope, reuse=reuse):
      embedded_inputs = self.embedding_layer.apply(seq_inputs, is_train)
      lstm_output_dic = self.lstm.apply(inputs=embedded_inputs, inputs_length=seq_lengths, is_train=is_train)

      with tf.variable_scope("OutputProjection", reuse=tf.AUTO_REUSE):
        logits = tf.contrib.layers.fully_connected(lstm_output_dic['sents_reps'],
                                                   activation_fn=None,
                                                   num_outputs=self.hparams.output_dim,
                                                   weights_initializer=self.output_fully_connected_weights,
                                                   biases_initializer=None)

      tf.logging.info("logits")
      tf.logging.info(logits)



      if self.hparams.output_dim > 1:
        predictions = tf.argmax(logits, axis=-1)
      else:
        predictions = tf.cast(tf.round(tf.nn.sigmoid(logits)), tf.int64)

      tf.logging.info("predictions")
      tf.logging.info(predictions)

      if self.hparams.output_dim > 1:
        loss = tf.reduce_mean(
          tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=tf.one_hot(labels, depth=2)))
      else:
        loss = tf.reduce_mean(
          tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.cast(tf.expand_dims(labels,axis=-1), dtype=tf.float32)))

      tf.logging.info("labels")
      tf.logging.info(labels)

      root_accuracy = tf.reduce_mean(tf.cast(tf.math.equal(predictions, tf.expand_dims(labels,axis=-1)), dtype=tf.float32))
      total_matchings = tf.reduce_sum(tf.cast(tf.math.equal(predictions, tf.expand_dims(labels,axis=-1)), dtype=tf.float32))

    return {'predictions': predictions,
            'logits': logits,
            'labels': labels,
            'loss': loss,
            'root_loss': loss,
            'root_accuracy': root_accuracy,
            'raw_outputs': lstm_output_dic['raw_outputs'],
            'embedded_inputs': embedded_inputs,
            'raw_inputs': seq_inputs,
            'total_matchings': total_matchings,
            'trainable_vars': tf.trainable_variables(scope=self.scope),
            'sents_reps': lstm_output_dic['sents_reps']}
