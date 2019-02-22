import tensorflow as tf
from tensorflow.python.ops import embedding_ops
from distill.layers.embedding import Embedding


class TreeLSTM(object):
  def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, input_keep_prob=0.8, hidden_keep_prob=0.8, depth=1, scope="TreeLSTM"):
    self.scope = scope
    self.hidden_dim = hidden_dim
    self.input_dim = input_dim
    self.embedding_dim = embedding_dim
    self.output_dim = output_dim
    self.input_keep_prob = input_keep_prob
    self.hidden_keep_prob = hidden_keep_prob
    self.num_layers = depth


  def create_vars(self, pretrained_word_embeddings, reuse=False):
    with tf.variable_scope(self.scope, reuse=reuse):
      self.embedding_layer = Embedding(vocab_size=self.input_dim,
                                       embedding_dim=self.embedding_dim,
                                       keep_prob=self.input_keep_prob)
      self.embedding_layer.create_vars(pretrained_word_embeddings)

      with tf.variable_scope('Composition'):
        self.W1 = tf.get_variable('W1',
                                  [2 * self.hidden_dim, self.hidden_dim])
        self.b1 = tf.get_variable('b1', [1, self.hidden_dim])

      # Build the RNN layers
      with tf.name_scope("LSTM_Cell"):
        lstm = tf.contrib.rnn.BasicLSTMCell(self.hidden_dim)
        lstm = tf.contrib.rnn.DropoutWrapper(lstm,
                                             output_keep_prob=self.hidden_keep_prob)
        self.multi_lstm_cell = lstm

      with tf.variable_scope('Projection'):
        self.U = tf.get_variable('U', [self.hidden_dim, self.output_dim])
        self.bs = tf.get_variable('bs', [1, self.output_dim])

        # Initialize the weights and biases
        self.input_fully_connected_weights = tf.truncated_normal_initializer(stddev=0.1)
        self.input_fully_connected_biases = tf.zeros_initializer()

        self.output_fully_connected_weights = tf.truncated_normal_initializer(stddev=0.1)
        self.output_fully_connected_biases = tf.zeros_initializer()


  def apply(self, examples):
    with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
      example_id, length, is_leaf, left_children, right_children, node_word_ids, labels, binary_labels, \
      root_label, root_binary_label, seq_lengths, seq_inputs = examples

      max_length = tf.reduce_max(length)
      output_tensor_array = tf.TensorArray(
        tf.float32,
        size=0,
        dynamic_size=True,
        clear_after_read=False,
        infer_shape=False)

      self.batch_size = left_children.get_shape()[0]
      output_tensor_array = output_tensor_array.write(0, tf.zeros((self.batch_size, self.hidden_dim),
                                                                  dtype=tf.float32))

      logit_tensor_array = tf.TensorArray(
        tf.float32,
        size=0,
        dynamic_size=True,
        clear_after_read=False,
        infer_shape=False)

      h_state_tensor_array = tf.TensorArray(
        tf.float32,
        size=0,
        dynamic_size=True,
        clear_after_read=False,
        infer_shape=False)

      c_state_tensor_array = tf.TensorArray(
        tf.float32,
        size=0,
        dynamic_size=True,
        clear_after_read=False,
        infer_shape=False)

      state = tf.nn.rnn_cell.LSTMStateTuple(tf.zeros((self.batch_size, self.hidden_dim)),
                                            tf.zeros((self.batch_size, self.hidden_dim)))

      h_state_tensor_array = h_state_tensor_array.write(0, state.h)
      c_state_tensor_array = c_state_tensor_array.write(0, state.c)

      def loop_body(tensor_array, logit_array, i, c_states, h_states,
                    is_leaf_placeholder, left_children_placeholder, right_children_placeholder,
                    node_word_indices_placeholder):
        node_is_leaf = is_leaf_placeholder[:, i]
        node_word_ids = node_word_indices_placeholder[:, i + 1]
        left_child = left_children_placeholder[:, i + 1]
        right_child = right_children_placeholder[:, i + 1]
        tf.logging.info(c_states.read(i))
        tf.logging.info(h_states.read(i))
        tf.logging.info(tf.cast(left_child, dtype=tf.int32))
        batch_indices = tf.expand_dims(tf.range(self.batch_size), 1)
        batch_indices = tf.concat([batch_indices, batch_indices], 1)
        tf.logging.info(batch_indices)
        left_indices = tf.cast(left_child, dtype=tf.int32)
        right_indices = tf.cast(right_child, dtype=tf.int32)

        tf.logging.info(left_indices)
        selected_left_tensor_array = tensor_array.gather(left_indices)
        selected_right_tensor_array = tensor_array.gather(right_indices)

        tf.logging.info(tf.gather_nd(selected_left_tensor_array, batch_indices))
        tf.logging.info(tf.gather_nd(selected_right_tensor_array, batch_indices))

        tf.logging.info("states")
        tf.logging.info(c_states.read(i))
        tf.logging.info(h_states.read(i))
        tf.logging.info(tf.nn.rnn_cell.LSTMStateTuple(c_states.read(i), h_states.read(i)))

        embedded_words, dummy_state = self.embed_word(node_word_ids)
        combined_nodes, combined_states = self.combine_children(tf.gather_nd(selected_left_tensor_array, batch_indices),
                                                                tf.gather_nd(selected_right_tensor_array, batch_indices),
                                                                c_states.read(i),
                                                                h_states.read(i))
        node_tensor = tf.where(
          tf.cast(node_is_leaf, tf.bool),
          embedded_words,
          combined_nodes
        )

        state_c = tf.where(
          tf.cast(node_is_leaf, tf.bool),
          dummy_state[0],
          combined_states[0]
        )

        state_h = tf.where(
          tf.cast(node_is_leaf, tf.bool),
          dummy_state[1],
          combined_states[1]
        )
        tf.logging.info(node_tensor)

        tf.logging.info("state")
        tf.logging.info(state)

        tensor_array = tensor_array.write(i + 1, node_tensor)
        logit_array = logit_array.write(i, tf.matmul(node_tensor, self.U) + self.bs)
        c_states = c_states.write(i + 1, state_c)
        h_states = h_states.write(i + 1, state_h)
        i = tf.add(i, 1)

        return tensor_array, logit_array, i, c_states, h_states, \
               is_leaf_placeholder, left_children_placeholder, right_children_placeholder, \
               node_word_indices_placeholder

      def loop_cond(unused_output_tensor_array, logit_array, i, unused_c_states_tensor_array,
                    unused_h_states_tensor_array,
                    is_leaf_placeholder, left_children_placeholder, right_children_placeholder,
                    node_word_indices_placeholder):
        """

        :param unused_output_tensor_array:
        :param i: counting number of nodes visited so far.
        :param unused_states_tensor_array:
        :return: True if i < total number of nodes
        """
        return tf.less(i, tf.cast(max_length, dtype=tf.int32))

      while_inputs = [output_tensor_array, logit_tensor_array, 0, c_state_tensor_array, h_state_tensor_array]

      while_inputs += [is_leaf,
                       left_children, right_children,
                       node_word_ids]

      self.output_tensor_array, self.logits_tensor_array, i, _, _, _, _, _, _ = tf.while_loop(loop_cond, loop_body,
                                                                                              while_inputs,
                                                                                              parallel_iterations=1)

      logits = tf.transpose(self.logits_tensor_array.stack(), [1, 0, 2])
      outputs = self.output_tensor_array.concat()[1:]

      bach_indices = tf.expand_dims(tf.range(self.batch_size), 1)
      root_indices = tf.concat([bach_indices, tf.expand_dims(tf.cast(length - 1, dtype=tf.int32), 1)], axis=-1)

      root_logits = tf.gather_nd(logits, root_indices)

      return {'outputs': outputs,
              'logits': logits,
              'root_logits': root_logits,
              'root_indices': root_indices}


  def combine_children(self, left_tensor, right_tensor, c_state, h_state):
    state = tf.nn.rnn_cell.LSTMStateTuple(c_state, h_state)
    output, state = self.multi_lstm_cell(
      inputs=tf.nn.relu(tf.matmul(tf.concat([left_tensor, right_tensor], axis=1), self.W1) + self.b1),
      state=state)

    tf.logging.info("lstm output")
    tf.logging.info(output)
    tf.logging.info(state.c)
    tf.logging.info(state.h)

    return output, state

  def embed_word(self, word_index):
    with tf.device('/cpu:0'):
      state = tf.nn.rnn_cell.LSTMStateTuple(tf.zeros((self.batch_size, self.hidden_dim)),
                                            tf.zeros((self.batch_size, self.hidden_dim)))
      tf.logging.info(word_index)
      embedded_words = tf.where(tf.less(word_index, 0),
                                tf.zeros((self.batch_size,
                                          self.embedding_layer.embedding_dim + self.embedding_layer.tuned_embedding_dim)),
                                self.embedding_layer.apply(word_index))

      # Create the fully connected layers
      with tf.variable_scope("Projection", reuse=tf.AUTO_REUSE):
        embedded_words = tf.contrib.layers.fully_connected(embedded_words,
                                                           num_outputs=self.hidden_dim,
                                                           weights_initializer=self.input_fully_connected_weights,
                                                           biases_initializer=self.input_fully_connected_biases)

      tf.logging.info("embedded words")
      tf.logging.info(embedded_words)
      tf.logging.info(state)
      return embedded_words, state
