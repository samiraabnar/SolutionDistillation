import tensorflow as tf
import numpy as np
from distill.layers.attention import FeedforwardSelfAttention

class LSTM(object):
  def __init__(self, hidden_dim, output_dim, attention_mechanism, hidden_keep_prob,depth, sent_rep_mode="all",scope="LSTM"):
    self.hidden_dim = hidden_dim
    self.output_dim = output_dim
    self.scope = scope
    self.hidden_keep_prob = hidden_keep_prob
    self.num_layers = depth
    self.attention_mechanism = attention_mechanism
    self.sent_rep_mode = sent_rep_mode
    self.sent_rep_dim = self.hidden_dim
    self.normalizer = tf.contrib.layers.layer_norm
    self.initializer = tf.contrib.layers.xavier_initializer()

  def create_vars(self, reuse=False, share_in_depth=True):
    with tf.variable_scope(self.scope, reuse=reuse):
      # Build the RNN layers
      with tf.variable_scope("LSTM_Cells"):
        lstm0 = tf.nn.rnn_cell.LSTMCell(self.hidden_dim, forget_bias=1.0,
                                        initializer=self.initializer, name="L0")
        dropout_lstm0 = tf.contrib.rnn.DropoutWrapper(lstm0,
                                                      output_keep_prob=self.hidden_keep_prob,
                                                      variational_recurrent=True,
                                                      dtype=tf.float32)

        lstms = [lstm0]
        drop_lstms = [dropout_lstm0]

        if self.num_layers > 1:
          if share_in_depth:
            lstm = tf.nn.rnn_cell.LSTMCell(self.hidden_dim, forget_bias=1.0,
                                           initializer=self.initializer, name="L1")
            dropout_lstm = tf.contrib.rnn.DropoutWrapper(lstm,
                                                          output_keep_prob=self.hidden_keep_prob,
                                                          variational_recurrent=True,
                                                          dtype=tf.float32
                                                         )
            lstms.extend([lstm] * (self.num_layers - 1))
            drop_lstms.extend([dropout_lstm] * (self.num_layers - 1))
          else:
            for i in np.arange(1, self.num_layers):
              lstm = tf.nn.rnn_cell.LSTMCell(self.hidden_dim, forget_bias=1.0,
                                             initializer=self.initializer, name="L"+str(i))
              dropout_lstm = tf.contrib.rnn.DropoutWrapper(lstm,
                                                           output_keep_prob=self.hidden_keep_prob,
                                                           variational_recurrent=True,
                                                           dtype=tf.float32
                                                           )
            lstms.extend([lstm] )
            drop_lstms.extend([dropout_lstm])

        self.multi_lstm_cell = tf.contrib.rnn.MultiRNNCell(lstms)
        self.multi_dropout_lstm_cell = tf.contrib.rnn.MultiRNNCell(drop_lstms)

      if self.attention_mechanism is not None:
        with tf.variable_scope("Attention"):
          self.attention = FeedforwardSelfAttention(scope="attention")
          self.attention.create_vars()

  def apply(self, inputs, inputs_length, init_state=None, is_train=True, cache=None):
    self.batch_size = tf.shape(inputs)[0]
    with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
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
        # lstm_outputs = tf_layers.layer_norm(lstm_outputs)


      if self.attention_mechanism is not None:
        with tf.variable_scope("Attention", reuse=tf.AUTO_REUSE):
          lstm_outputs = self.attention.apply(lstm_outputs, inputs_length, is_train)


      bach_indices = tf.expand_dims(tf.range(self.batch_size), 1)
      root_indices = tf.concat([bach_indices, tf.expand_dims(tf.cast(inputs_length - 1, dtype=tf.int32), 1)], axis=-1)

      inputs_mask = tf.expand_dims(tf.cast(tf.sequence_mask(inputs_length), tf.float32),-1)

      if self.sent_rep_mode == "all":
        sentence_reps = tf.reduce_sum(lstm_outputs * inputs_mask, axis=1) / tf.expand_dims(tf.cast(inputs_length, tf.float32), -1)
      elif self.sent_rep_mode == "final":
        sentence_reps = tf.gather_nd(lstm_outputs, root_indices)
      else:
        sentence_reps = tf.reduce_sum(lstm_outputs * inputs_mask, axis=1)


    return {
            'raw_outputs': lstm_outputs,
            'sents_reps': sentence_reps,
            'seq_outputs': lstm_outputs,
            'final_state': final_state,
            'outputs_lengths': inputs_length
    }
    
  def get_the_cell(self, is_train=True):
    # Run the data through the RNN layers
    with tf.variable_scope("LSTM_Cells", reuse=tf.AUTO_REUSE):
      the_cell = self.multi_lstm_cell
      if is_train:
        the_cell = self.multi_dropout_lstm_cell
        
    return the_cell

  def predict(self, compute_decoding_step_input_fn, inputs_length, input_embedding_layer, output_embedding_layer, eos_id,
              target_length=None, init_state=None, is_train=False, initial_inputs=None):
    batch_size = tf.shape(inputs_length)[0]
    with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):

      # Choose the cell with or without dropout
      the_cell = self.multi_lstm_cell
      if is_train:
        the_cell = self.multi_dropout_lstm_cell

      # Set maximum sequence Length
      if target_length is None:
        timesteps = tf.reduce_max(inputs_length)
      else:
        timesteps = target_length

      # TensorArray to keep final LSTM outputs
      all_outputs_tensor_array = tf.TensorArray(dtype=tf.float32, size=0,
                                   dynamic_size=True,
                                   clear_after_read=False,
                                   infer_shape=True)

      # TensorArray to keep predictions
      sampled_predictions_tensor_array = tf.TensorArray(dtype=tf.int64, size=0,
                                                dynamic_size=True,
                                                clear_after_read=False,
                                                infer_shape=True)

      # Loop condition
      for_each_time_step = lambda sp, lp, ls, o, f, step: tf.logical_and(
        tf.less(tf.cast(step, dtype=tf.int32), tf.cast(timesteps, dtype=tf.int32)),
        tf.logical_not(tf.reduce_all(f)))

      # This loop gets called once for every "timestep" and obtains one column of the input data
      def lstm_loop(sampled_predictions,
                    last_prediction, last_state,
                    output_lengths, finish_flags, step):
        """

        :param sampled_predictions: tensor array to save predictions
        :param last prediction: [batch_size,1,1]
        :param last_state:  [batch_size, hidden_dim]
        :param output_lengths: [batch size, 1]
        :param finish_flags: [batch_size,1]
        :param step: scalar integer
        :return:
        """

        last_prediction = tf.squeeze(last_prediction)
        embedded_prediction = input_embedding_layer.apply(last_prediction)

        current_step_input = compute_decoding_step_input_fn(embedded_prediction)
        if current_step_input is not None:
          cell_input = tf.concat([embedded_prediction, current_step_input], axis=-1)
        else:
          cell_input = embedded_prediction

        tf.logging.info("cell inputs")
        tf.logging.info(cell_input)
        lstm_outputs, state = the_cell(cell_input, last_state)
        lstm_outputs = lstm_outputs[:,None,:]
        tf.logging.info("cell outputs")
        tf.logging.info(lstm_outputs)
        tf.logging.info(state)
        logits = output_embedding_layer(lstm_outputs)
        prediction = tf.random.multinomial(logits=tf.squeeze(logits),
                                           num_samples=1)
        tf.logging.info("prediction shape")
        tf.logging.info(prediction)
        sampled_predictions.write(step, prediction)

        finish_flags = tf.logical_or(finish_flags,tf.equal(prediction[:,-1],eos_id))
        output_lengths = output_lengths + tf.cast( tf.logical_not(finish_flags), dtype=tf.int32)*1

        return sampled_predictions, prediction, state, output_lengths, finish_flags, tf.add(step, 1)



      # Initial outpus are zero unless initial inputs are not None (e.g. we have a start of sentence token)
      # initial_outputs = tf.zeros([batch_size, self.hidden_dim])
      # if initial_inputs is not None:
      #   initial_outputs = output_embedding_layer.apply(initial_inputs)

      # Initially none of the sentences are finished
      init_finish = tf.cast(tf.zeros(batch_size, dtype=tf.int64), dtype=tf.bool)

      #Initially generated sentence lengths are all zero
      init_output_lengths = tf.zeros(batch_size, dtype=tf.int32)

      # If initial state is None set it to zero
      if init_state is None:
        init_state = the_cell.zero_state(batch_size, tf.float32)

      samples, final_prediction, lstm_state, output_lengths, _, _ = tf.while_loop(for_each_time_step,
                                                                                               lstm_loop,
                                                                                               (sampled_predictions_tensor_array,
                                                                                                initial_inputs,
                                                                                                init_state,
                                                                                                init_output_lengths,
                                                                                                init_finish,
                                                                                                0),
                                                                                                parallel_iterations=32)


      samples = tf.transpose(samples.stack(), [1, 0, 2])


    return {
            'final_state': lstm_state,
            'outputs_lengths': output_lengths,
            'samples': samples
    }



