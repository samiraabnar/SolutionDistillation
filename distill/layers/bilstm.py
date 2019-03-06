import tensorflow as tf
import tensorflow.contrib.layers as tf_layers
from distill.layers.attention import FeedforwardSelfAttention
from distill.layers.embedding import Embedding

class BiLSTM(object):
  def __init__(self, input_dim, hidden_dim, output_dim, attention_mechanism=None, input_keep_prob=0.8, hidden_keep_prob=0.8,depth=1, scope="LSTM"):
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.output_dim = output_dim
    self.scope = scope
    self.input_keep_prob = input_keep_prob
    self.hidden_keep_prob = hidden_keep_prob
    self.num_layers = depth
    self.attention_mechanism = attention_mechanism

  def create_vars(self, pretrained_word_embeddings, reuse=False):
    # Create the embeddings
    with tf.variable_scope(self.scope, reuse=reuse):
      self.embedding_layer = Embedding(vocab_size=self.input_dim, keep_prob=self.input_keep_prob)
      self.embedding_layer.create_vars(pretrained_word_embeddings)

      # Build the RNN layers
      with tf.variable_scope("LSTM_Cell"):
        lstm = tf.contrib.rnn.BasicLSTMCell(self.hidden_dim, forget_bias=1.0)
        dropout_lstm = tf.contrib.rnn.DropoutWrapper(lstm,
                                               output_keep_prob=self.hidden_keep_prob)

        self.fw_multi_lstm_cell = tf.contrib.rnn.MultiRNNCell([lstm] * self.num_layers)
        self.fw_multi_dropout_lstm_cell = tf.contrib.rnn.MultiRNNCell([dropout_lstm] * self.num_layers)

        lstm = tf.contrib.rnn.BasicLSTMCell(self.hidden_dim)
        dropout_lstm = tf.contrib.rnn.DropoutWrapper(lstm,
                                               output_keep_prob=self.hidden_keep_prob)
        self.bw_multi_lstm_cell = tf.contrib.rnn.MultiRNNCell([lstm] * self.num_layers)
        self.bw_multi_dropout_lstm_cell = tf.contrib.rnn.MultiRNNCell([dropout_lstm] * self.num_layers)

      if self.attention_mechanism is not None:
        with tf.variable_scope("Attention"):
          self.attention = FeedforwardSelfAttention(scope="attention")
          self.attention.create_vars()

        # Create the fully connected layers
      with tf.variable_scope("Projection"):
        # Initialize the weights and biases
        self.input_fully_connected_weights = tf.contrib.layers.xavier_initializer()

        self.output_fully_connected_weights = tf.contrib.layers.xavier_initializer()

  def apply(self, inputs, inputs_length, is_train=True):
    self.batch_size = inputs.get_shape()[0]
    with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
      embedded_input = self.embedding_layer.apply(inputs, is_train)
      #embedded_input = tf_layers.layer_norm(embedded_input)
      tf.logging.info("embedded_input")
      tf.logging.info(embedded_input)

      # Create the fully connected layers
      with tf.variable_scope("InputProjection", reuse=tf.AUTO_REUSE):
        embedded_input = tf.contrib.layers.fully_connected(embedded_input,
                                                   num_outputs=self.hidden_dim,
                                                   weights_initializer=self.input_fully_connected_weights,
                                                   biases_initializer=None)


      # Run the data through the RNN layers
      with tf.variable_scope("LSTM_Cell", reuse=tf.AUTO_REUSE):
        fw_cell = self.fw_multi_lstm_cell
        bw_cell = self.bw_multi_lstm_cell
        if is_train:
          fw_cell = self.fw_multi_dropout_lstm_cell
          bw_cell = self.bw_multi_dropout_lstm_cell

        lstm_outputs, final_state = tf.nn.bidirectional_dynamic_rnn(
          cell_fw=fw_cell,
          cell_bw=bw_cell,
          inputs=embedded_input,
          sequence_length=inputs_length,
          dtype=tf.float32,
        )

        #lstm_outputs = tf_layers.layer_norm(lstm_outputs)




      # concatenation output from forward and backward layers.
      fw_outputs, bw_outputs = tf.unstack(lstm_outputs)
      lstm_outputs = tf.concat([fw_outputs, bw_outputs], axis=-1)

      tf.logging.info("after concat")
      tf.logging.info(lstm_outputs)
      bach_indices = tf.expand_dims(tf.range(self.batch_size), 1)
      root_indices = tf.concat([bach_indices, tf.expand_dims(tf.cast(inputs_length - 1, dtype=tf.int32), 1)], axis=-1)

      if self.attention_mechanism is not None:
        with tf.variable_scope("Attention", reuse=tf.AUTO_REUSE):
          lstm_outputs = self.attention.apply(lstm_outputs, is_train)

      tf.logging.info("LSTM output before projection")
      tf.logging.info(lstm_outputs)
      tf.logging.info(inputs_length)

      # Sum over all representations for each sentence!
      inputs_mask = tf.expand_dims(tf.cast(tf.sequence_mask(inputs_length), tf.float32), -1)
      sentence_reps = tf.reduce_sum(lstm_outputs * inputs_mask, axis=1)

      # Create the fully connected layers
      with tf.variable_scope("OutputProjection", reuse=tf.AUTO_REUSE):
        logits = tf.contrib.layers.fully_connected(sentence_reps,
                                                    num_outputs=self.output_dim,
                                                    weights_initializer=self.output_fully_connected_weights,
                                                    biases_initializer=None)

    return {'logits': logits,
            'raw_outputs': lstm_outputs,
            'embedded_inputs': embedded_input,
            'raw_inputs': inputs,
    }




