import tensorflow as tf
import tensorflow.contrib.layers as tf_layers
from distill.layers.attention import FeedforwardSelfAttention
from distill.layers.embedding import Embedding

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

  def create_vars(self, reuse=False):
    with tf.variable_scope(self.scope, reuse=reuse):
      # Build the RNN layers
      with tf.variable_scope("LSTM_Cells"):
        lstm0 = tf.nn.rnn_cell.LSTMCell(self.hidden_dim, forget_bias=1.0, name="L0")
        dropout_lstm0 = tf.contrib.rnn.DropoutWrapper(lstm0,
                                                      output_keep_prob=self.hidden_keep_prob,
                                                      variational_recurrent=True,
                                                      dtype=tf.float32)

        lstms = [lstm0]
        drop_lstms = [dropout_lstm0]

        lstm = tf.nn.rnn_cell.LSTMCell(self.hidden_dim, forget_bias=1.0, name="L1")
        dropout_lstm = tf.contrib.rnn.DropoutWrapper(lstm,
                                                      output_keep_prob=self.hidden_keep_prob,
                                                      variational_recurrent=True,
                                                      dtype=tf.float32
                                                     )
        if self.num_layers > 1:
          lstms.extend([lstm] * (self.num_layers-1))
          drop_lstms.extend([dropout_lstm] * (self.num_layers - 1))

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

      # Sum over all representations for each sentence!
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

  def predict(self, compute_decoding_step_input_fn, inputs_length, embedding_layer, eos_id,
              target_length=None, init_state=None, is_train=True, initial_inputs=None):
    batch_size = tf.shape(inputs_length)[0]
    with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
      # Run the data through the RNN layers
      with tf.variable_scope("LSTM_Cells", reuse=tf.AUTO_REUSE):
        the_cell = self.multi_lstm_cell
        if is_train:
          the_cell = self.multi_dropout_lstm_cell

        if init_state is None:
          init_state= the_cell.zero_state(batch_size, tf.float32)

        all_outputs_tensor_array = tf.TensorArray(dtype=tf.float32, size=0,
                                     dynamic_size=True,
                                     clear_after_read=False,
                                     infer_shape=True)

        # This loop gets called once for every "timestep" and obtains one column of the input data
        def lstm_loop(output_lengths, all_outputs, last_lstm_prediction,last_state, finish_flags, step):


          last_lstm_prediction_logits = tf.expand_dims(last_lstm_prediction, 1)
          last_lstm_prediction_logits = embedding_layer.linear(last_lstm_prediction_logits)
          prediction = tf.random.multinomial(logits=tf.squeeze(last_lstm_prediction_logits),
                                             num_samples=1)
          embedded_prediction = embedding_layer.apply(prediction)
          embedded_prediction = embedded_prediction[:,-1,:]

          current_step_input = compute_decoding_step_input_fn(embedded_prediction)
          if current_step_input is not None:
            cell_input = tf.concat([embedded_prediction, current_step_input], axis=-1)
          else:
            cell_input = embedded_prediction

          lstm_prediction, state = the_cell(cell_input, last_state)

          all_outputs = all_outputs.write(step, lstm_prediction)
          finish_flags = tf.logical_or(finish_flags,tf.equal(prediction[:,-1],eos_id))
          output_lengths = output_lengths + tf.cast( tf.logical_not(finish_flags), dtype=tf.int32)*1

          return output_lengths, all_outputs, lstm_prediction, state, finish_flags, tf.add(step, 1)


        if target_length is None:
          timesteps = tf.reduce_max(inputs_length)
        else:
          timesteps = target_length

        for_each_time_step = lambda l, c, a, b, f, step: tf.logical_and(
          tf.less(tf.cast(step, dtype=tf.int32), tf.cast(timesteps, dtype=tf.int32)),
          tf.logical_not(tf.reduce_all(f)))

        initial_outputs = tf.zeros([batch_size, self.hidden_dim])
        if initial_inputs is not None:
          initial_outputs = initial_inputs
        init_finish = tf.cast(tf.zeros(batch_size, dtype=tf.int64), dtype=tf.bool)
        init_output_lengths = tf.zeros(batch_size, dtype=tf.int32)
        output_lengths, all_outputs, final_prediction, lstm_state, _, _ = tf.while_loop(for_each_time_step, lstm_loop,
                                                       (init_output_lengths, all_outputs_tensor_array,
                                                        initial_outputs, init_state,init_finish, 0),
                                                       parallel_iterations=32)


        lstm_outputs = tf.transpose(all_outputs.stack(),[1,0,2])


      if self.attention_mechanism is not None:
        with tf.variable_scope("Attention", reuse=tf.AUTO_REUSE):
          lstm_outputs = self.attention.apply(lstm_outputs, inputs_length, is_train)


      bach_indices = tf.expand_dims(tf.range(batch_size), 1)
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




    return {
            'raw_outputs': lstm_outputs,
            'sents_reps': sentence_reps,
            'seq_outputs': lstm_outputs,
            'final_state': lstm_state,
            'outputs_lengths': output_lengths
    }


