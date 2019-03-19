import tensorflow as tf
import tensorflow.contrib.layers as tf_layers
from distill.layers.attention import FeedforwardSelfAttention
from distill.layers.embedding import Embedding

class LSTM(object):
  def __init__(self, hidden_dim, output_dim, attention_mechanism=None, hidden_keep_prob=0.8,depth=1, sent_rep_mode="all",scope="LSTM"):
    self.hidden_dim = hidden_dim
    self.output_dim = output_dim
    self.scope = scope
    self.hidden_keep_prob = hidden_keep_prob
    self.num_layers = depth
    self.attention_mechanism = attention_mechanism
    self.sent_rep_mode = sent_rep_mode

  def create_vars(self, reuse=False):
    # Create the embeddings
    with tf.variable_scope(self.scope, reuse=reuse):
      # Build the RNN layers
      with tf.variable_scope("LSTM_Cells"):
        lstm0 = tf.nn.rnn_cell.LSTMCell(self.hidden_dim, forget_bias=1.0, name="L0")
        dropout_lstm0 = tf.contrib.rnn.DropoutWrapper(lstm0,
                                               output_keep_prob=self.hidden_keep_prob)

        lstms = [lstm0]
        drop_lstms = [dropout_lstm0]

        lstm = tf.nn.rnn_cell.LSTMCell(self.hidden_dim, forget_bias=1.0, name="L1")
        dropout_lstm = tf.contrib.rnn.DropoutWrapper(lstm,
                                                      output_keep_prob=self.hidden_keep_prob)
        if self.num_layers > 1:
          lstms.extend([lstm] * (self.num_layers-1))
          drop_lstms.extend([dropout_lstm] * (self.num_layers - 1))

        self.multi_lstm_cell = tf.contrib.rnn.MultiRNNCell(lstms)
        self.multi_dropout_lstm_cell = tf.contrib.rnn.MultiRNNCell(drop_lstms)

      if self.attention_mechanism is not None:
        with tf.variable_scope("Attention"):
          self.attention = FeedforwardSelfAttention(scope="attention")
          self.attention.create_vars()



  def apply(self, inputs, inputs_length, is_train=True):
    self.batch_size = inputs.get_shape()[0]
    with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
      #embedded_input = tf_layers.layer_norm(embedded_input)
      tf.logging.info("embedded_input")
      tf.logging.info(inputs)

      #embedded_input = tf_layers.layer_norm(embedded_input)

      # Create the fully connected layers
      # with tf.variable_scope("InputProjection", reuse=tf.AUTO_REUSE):
      #   embedded_input = tf.contrib.layers.fully_connected(embedded_input,
      #                                              num_outputs=self.hidden_dim,
      #                                              weights_initializer=self.input_fully_connected_weights,
      #                                              biases_initializer=None)


      # Run the data through the RNN layers
      with tf.variable_scope("LSTM_Cell", reuse=tf.AUTO_REUSE):
        cell = self.multi_lstm_cell
        if is_train:
          cell = self.multi_dropout_lstm_cell

        lstm_outputs, final_state = tf.nn.dynamic_rnn(
          cell,
          inputs,
          dtype=tf.float32,
          sequence_length=inputs_length)
        #lstm_outputs = tf_layers.layer_norm(lstm_outputs)

      if self.attention_mechanism is not None:
        with tf.variable_scope("Attention", reuse=tf.AUTO_REUSE):
          lstm_outputs = self.attention.apply(lstm_outputs, inputs_length, is_train)


      tf.logging.info("LSTM output before projection")
      tf.logging.info(lstm_outputs)
      tf.logging.info(inputs_length)

      bach_indices = tf.expand_dims(tf.range(self.batch_size), 1)
      root_indices = tf.concat([bach_indices, tf.expand_dims(tf.cast(inputs_length - 1, dtype=tf.int32), 1)], axis=-1)

      # Sum over all representations for each sentence!
      inputs_mask = tf.expand_dims(tf.cast(tf.sequence_mask(inputs_length), tf.float32),-1)
      # sentence_reps = tf.gather_nd(lstm_outputs, root_indices)#tf.reduce_sum(lstm_outputs * inputs_mask, axis=1)
      if self.sent_rep_mode == "all":
        sentence_reps = tf.reduce_sum(lstm_outputs * inputs_mask, axis=1) / tf.expand_dims(tf.cast(inputs_length, tf.float32), -1)
      elif self.sent_rep_mode == "final":
        sentence_reps = tf.gather_nd(lstm_outputs, root_indices)
      else:
        sentence_reps = tf.reduce_sum(lstm_outputs * inputs_mask, axis=1)

      tf.logging.info("final output:")
      tf.logging.info(sentence_reps)



    return {
            'raw_outputs': lstm_outputs,
            'sents_reps': sentence_reps,
    }







