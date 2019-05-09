import tensorflow as tf
import tensorflow.contrib.layers as tf_layers
from distill.layers.attention import FeedforwardSelfAttention
from distill.layers.embedding import Embedding

class BiLSTM(object):
  def __init__(self, hidden_dim, output_dim, attention_mechanism,
               hidden_keep_prob,depth, sent_rep_mode, scope="biLSTM"):
    self.hidden_dim = hidden_dim
    self.output_dim = output_dim
    self.scope = scope
    self.hidden_keep_prob = hidden_keep_prob
    self.num_layers = depth
    self.attention_mechanism = attention_mechanism
    self.sent_rep_mode = sent_rep_mode
    self.sent_rep_dim = self.hidden_dim * 2

  def create_vars(self, reuse=False):
    # Create the embeddings
    with tf.variable_scope(self.scope, reuse=reuse):
      # Build the RNN layers
      with tf.variable_scope("LSTM_Cell"):
        fw_lstm_0 = tf.contrib.rnn.BasicLSTMCell(self.hidden_dim, forget_bias=1.0)
        dropout_fw_lstm_0 = tf.contrib.rnn.DropoutWrapper(fw_lstm_0,
                                               output_keep_prob=self.hidden_keep_prob)
        fw_lstm_1 = tf.contrib.rnn.BasicLSTMCell(self.hidden_dim, forget_bias=1.0)
        dropout_fw_lstm_1 = tf.contrib.rnn.DropoutWrapper(fw_lstm_1,
                                                       output_keep_prob=self.hidden_keep_prob)

        self.fw_multi_lstm_cell = tf.contrib.rnn.MultiRNNCell([fw_lstm_0]+[fw_lstm_1] * (self.num_layers - 1))
        self.fw_multi_dropout_lstm_cell = tf.contrib.rnn.MultiRNNCell([dropout_fw_lstm_0]+[dropout_fw_lstm_1] * (self.num_layers - 1))

        bw_lstm_0 = tf.contrib.rnn.BasicLSTMCell(self.hidden_dim, forget_bias=1.0)
        dropout_bw_lstm_0 = tf.contrib.rnn.DropoutWrapper(bw_lstm_0,
                                                          output_keep_prob=self.hidden_keep_prob)
        bw_lstm_1 = tf.contrib.rnn.BasicLSTMCell(self.hidden_dim, forget_bias=1.0)
        dropout_bw_lstm_1 = tf.contrib.rnn.DropoutWrapper(bw_lstm_1,
                                                          output_keep_prob=self.hidden_keep_prob)

        self.bw_multi_lstm_cell = tf.contrib.rnn.MultiRNNCell([bw_lstm_0] + [bw_lstm_1] * (self.num_layers - 1))
        self.bw_multi_dropout_lstm_cell = tf.contrib.rnn.MultiRNNCell(
          [dropout_bw_lstm_0] + [dropout_bw_lstm_1] * (self.num_layers - 1))

    if self.attention_mechanism is not None:
        with tf.variable_scope("Attention"):
          self.attention = FeedforwardSelfAttention(scope="attention")
          self.attention.create_vars()

  def apply(self, inputs, inputs_length, init_state=None, is_train=True):
    self.batch_size = tf.shape(inputs)[0]
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
          sequence_length=tf.cast(inputs_length, dtype=tf.int32),
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
            'outputs_lengths': inputs_length,
    }

  def predict(self, inputs_length,
              compute_decoding_step_input_fn,
              embedding_layer, eos_id, output_embedding_fn=None, target_length=None, init_state=None, is_train=True):
    self.batch_size = tf.shape(inputs_length)[0]
    with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
      # Run the data through the RNN layers
      with tf.variable_scope("LSTM_Cell", reuse=tf.AUTO_REUSE):
        fw_cell = self.fw_multi_lstm_cell
        bw_cell = self.bw_multi_lstm_cell
        if is_train:
          fw_cell = self.fw_multi_dropout_lstm_cell
          bw_cell = self.bw_multi_dropout_lstm_cell

        if init_state is None:
          init_state= (fw_cell.zero_state(self.batch_size, tf.float32), bw_cell.zero_state(self.batch_size, tf.float32))

        all_outputs_tensor_array = tf.TensorArray(dtype=tf.float32, size=0,
                                     dynamic_size=True,
                                     clear_after_read=False,
                                     infer_shape=True)

        # This loop gets called once for every "timestep" and obtains one column of the input data
        def lstm_loop(output_lengths, all_outputs, last_lstm_prediction, last_state,finish_flags, step):
          if output_embedding_fn is not None:
            last_lstm_prediction = output_embedding_fn(last_lstm_prediction)

          last_lstm_prediction = tf.expand_dims(last_lstm_prediction, 1)
          logits = embedding_layer.linear(last_lstm_prediction)
          tf.logging.info('logits in the loop')
          tf.logging.info(logits) #(batch_size, 1, 11)
          prediction = tf.argmax(logits, axis=-1) #(batch_size,1)
          tf.logging.info(prediction)
          embedded_prediction = embedding_layer.apply(prediction) #(batch_size,1, embedding_dim)
          embedded_prediction = embedded_prediction[:, -1, :] #(batch_size, embedding_dim)
          step_input = compute_decoding_step_input_fn(embedded_prediction)
          cell_input = tf.expand_dims(tf.concat([embedded_prediction, step_input], axis=-1), axis=1)

          tf.logging.info(cell_input)
          lstm_outs, state = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell,
                                                                   cell_bw=bw_cell,
                                                                   inputs=cell_input,
                                                                   dtype=tf.float32)

          tf.logging.info(state)
          # concatenation output from forward and backward layers.
          fw_outputs, bw_outputs = tf.unstack(lstm_outs)
          lstm_prediction = tf.concat([fw_outputs[:,-1,:], bw_outputs[:,-1,:]], axis=-1)
          tf.logging.info('lstm_prediction')
          tf.logging.info(lstm_prediction)
          all_outputs = all_outputs.write(step, lstm_prediction)

          finish_flags = tf.logical_or(finish_flags, tf.equal(prediction[:, -1], eos_id))
          output_lengths = output_lengths + tf.cast(tf.logical_not(finish_flags), dtype=tf.int32) * 1

          return output_lengths, all_outputs, lstm_prediction, state, finish_flags, tf.add(step, 1)

        if target_length is None:
          timesteps = tf.reduce_max(inputs_length)
        else:
          timesteps = target_length

        for_each_time_step = lambda l, c, a, b, f, step: tf.logical_and(tf.less(tf.cast(step, dtype=tf.int32),
                                                        tf.cast(timesteps, dtype=tf.int32)),
                                                            tf.logical_not(tf.reduce_all(f)))

        initial_prediction = tf.zeros([self.batch_size, self.hidden_dim*2])
        init_finish = tf.cast(tf.zeros(self.batch_size, dtype=tf.int64), dtype=tf.bool)
        init_output_lengths = tf.zeros(self.batch_size, dtype=tf.int32)
        output_lengths, all_outputs, final_prediction, lstm_state, _, _ = tf.while_loop(for_each_time_step, lstm_loop,
                                                       (init_output_lengths, all_outputs_tensor_array,
                                                        initial_prediction, init_state,init_finish, 0),
                                                       parallel_iterations=32)


        lstm_outputs = tf.transpose(all_outputs.stack(),[1,0,2])
        tf.logging.info("while loop seq_outputs")
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
            'final_state': lstm_state,
            'outputs_lengths': output_lengths

    }


