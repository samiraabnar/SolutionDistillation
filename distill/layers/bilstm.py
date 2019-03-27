import tensorflow as tf
import tensorflow.contrib.layers as tf_layers
from distill.layers.attention import FeedforwardSelfAttention
from distill.layers.embedding import Embedding

class BiLSTM(object):
  def __init__(self, hidden_dim, output_dim, attention_mechanism=None, input_keep_prob=0.8,
               hidden_keep_prob=0.8,depth=1, sent_rep_mode="all", scope="biLSTM"):
    self.hidden_dim = hidden_dim
    self.output_dim = output_dim
    self.scope = scope
    self.input_keep_prob = input_keep_prob
    self.hidden_keep_prob = hidden_keep_prob
    self.num_layers = depth
    self.attention_mechanism = attention_mechanism
    self.sent_rep_mode = sent_rep_mode

  def create_vars(self, reuse=False):
    # Create the embeddings
    with tf.variable_scope(self.scope, reuse=reuse):
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


  def apply(self, inputs, inputs_length, is_train=True):
    self.batch_size = inputs.get_shape()[0]
    with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
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
          inputs=inputs,
          sequence_length=inputs_length,
          dtype=tf.float32,
        )

      # concatenation output from forward and backward layers.
      fw_outputs, bw_outputs = tf.unstack(lstm_outputs)
      lstm_outputs = tf.concat([fw_outputs, bw_outputs], axis=-1)

      tf.logging.info("after concat")
      tf.logging.info(lstm_outputs)
      bach_indices = tf.expand_dims(tf.range(self.batch_size), 1)
      root_indices = tf.concat([bach_indices, tf.expand_dims(tf.cast(inputs_length - 1, dtype=tf.int32), 1)], axis=-1)

      if self.attention_mechanism is not None:
        with tf.variable_scope("Attention", reuse=tf.AUTO_REUSE):
          lstm_outputs = self.attention.apply(lstm_outputs, inputs_length, is_train)

      tf.logging.info("LSTM output before projection")
      tf.logging.info(lstm_outputs)
      tf.logging.info(inputs_length)

      inputs_mask = tf.expand_dims(tf.cast(tf.sequence_mask(inputs_length), tf.float32), -1)
      if self.sent_rep_mode == "all": # Sum over all representations for each sentence!
        sentence_reps = tf.reduce_sum(lstm_outputs * inputs_mask, axis=1) / tf.expand_dims(tf.cast(inputs_length, tf.float32), -1)
      elif self.sent_rep_mode == "final":
        fw_sentence_reps = tf.gather_nd(fw_outputs, root_indices)
        bw_sentence_reps = bw_outputs[:,-1]
        sentence_reps = tf.concat([fw_sentence_reps,bw_sentence_reps], axis=-1)
      else:
        sentence_reps = tf.reduce_sum(lstm_outputs * inputs_mask, axis=1)


    return {
            'raw_outputs': lstm_outputs,
            'sents_reps': sentence_reps,
            'seq_outputs': lstm_outputs,
    }




