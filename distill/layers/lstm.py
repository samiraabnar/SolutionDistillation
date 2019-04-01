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
    self.sent_rep_dim = self.hidden_dim

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



  def apply(self, inputs, inputs_length, init_state=None, is_train=True):
    self.batch_size = tf.shape(inputs)[0]
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
      with tf.variable_scope("LSTM_Cells", reuse=tf.AUTO_REUSE):
        cell = self.multi_lstm_cell
        if is_train:
          cell = self.multi_dropout_lstm_cell

        if init_state is None:
          init_state= cell.zero_state(self.batch_size, tf.float32)
        lstm_outputs, final_state = tf.nn.dynamic_rnn(
          cell,
          inputs,
          dtype=tf.float32,
          sequence_length=inputs_length,
          initial_state=init_state)
        #lstm_outputs = tf_layers.layer_norm(lstm_outputs)

        tf.logging.info("seq_outputs"),
        tf.logging.info(lstm_outputs)

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
            'seq_outputs': lstm_outputs,
            'final_state': final_state
    }



  def infer_apply(self, inputs, inputs_length, output_embedding_fn, init_state=None, is_train=True):
    self.batch_size = tf.shape(inputs)[0]
    with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
      tf.logging.info("embedded_input")
      tf.logging.info(inputs)

      # Run the data through the RNN layers
      with tf.variable_scope("LSTM_Cells", reuse=tf.AUTO_REUSE):
        the_cell = self.multi_lstm_cell
        if is_train:
          the_cell = self.multi_dropout_lstm_cell

        if init_state is None:
          init_state= the_cell.zero_state(self.batch_size, tf.float32)

        all_outputs_tensor_array = tf.TensorArray(dtype=tf.float32, size=0,
                                     dynamic_size=True,
                                     clear_after_read=False,
                                     infer_shape=True)

        # This loop gets called once for every "timestep" and obtains one column of the input data
        def lstm_loop(all_outputs, last_lstm_prediction, last_state, step):
          tf.logging.info("last state")
          tf.logging.info(last_state)

          tf.logging.info(inputs[:, step, :])
          last_lstm_prediction = output_embedding_fn(last_lstm_prediction)

          tf.logging.info(last_lstm_prediction)
          cell_input = tf.concat([last_lstm_prediction, inputs[:, step, :]], axis=-1)
          tf.logging.info(cell_input)
          lstm_prediction, state = the_cell(cell_input, last_state)
          all_outputs = all_outputs.write(step, lstm_prediction)
          return all_outputs, lstm_prediction, state, tf.add(step, 1)

        initial_prediction = tf.zeros([self.batch_size, self.hidden_dim])

        timesteps = tf.reduce_max(inputs_length)

        for_each_time_step = lambda c, a, b, step: tf.less(tf.cast(step, dtype=tf.int32),
                                                        tf.cast(timesteps, dtype=tf.int32))

        all_outputs, final_output, lstm_state, _ = tf.while_loop(for_each_time_step, lstm_loop,
                                                       (all_outputs_tensor_array,
                                                        initial_prediction, init_state, 0),
                                                       parallel_iterations=32)

        lstm_outputs = all_outputs.stack()
        tf.logging.info("seq_outputs"),
        tf.logging.info(lstm_outputs)

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
            'seq_outputs': lstm_outputs,
            'final_state': lstm_state
    }





