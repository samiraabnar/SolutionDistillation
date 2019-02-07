import tensorflow as tf

class LSTM(object):
  def __init__(self, input_dim, hidden_dim, output_dim, keep_prob, num_layers=1, scope="LSTM"):
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.output_dim = output_dim
    self.scope = scope

    self.keep_prob = keep_prob
    self.num_layers = num_layers


  def create_vars(self, reuse=False):
    # Create the embeddings
    with tf.variable_scope(self.scope, reuse=reuse):
      with tf.variable_scope("Embeddings"):
        self.embedding = tf.Variable(tf.random_uniform((self.input_dim,
                                                   self.hidden_dim ), -1, 1))

      # Build the RNN layers
      with tf.name_scope("LSTM_Cell"):
        lstm = tf.contrib.rnn.BasicLSTMCell(self.hidden_dim)
        drop = tf.contrib.rnn.DropoutWrapper(lstm,
                                             output_keep_prob=self.keep_prob)
        self.multi_lstm_cell = tf.contrib.rnn.MultiRNNCell([drop] * self.num_layers)



        # Create the fully connected layers
      with tf.variable_scope("Projection"):
        # Initialize the weights and biases
        self.fully_connected_weights = tf.truncated_normal_initializer(stddev=0.1)
        self.fully_connected_biases = tf.zeros_initializer()



  def apply(self, inputs):
    with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
      with tf.variable_scope("Embedding"):
        embedded_input = tf.nn.embedding_lookup(self.embedding, inputs)


    # Run the data through the RNN layers
    with tf.variable_scope("LSTM_Cell", reuse=tf.AUTO_REUSE):
      lstm_outputs, final_state = tf.nn.dynamic_rnn(
        self.multi_lstm_cell,
        embedded_input)

      # Create the fully connected layers
    with tf.variable_scope("Projection", reuse=tf.AUTO_REUSE):
      logits = tf.contrib.layers.fully_connected(lstm_outputs[:, -1],
                                                  num_outputs=self.output_dim,
                                                  activation_fn=tf.linear,
                                                  weights_initializer=self.fully_connected_weights,
                                                  biases_initializer=self.fully_connected_biases)

    return {'logits': logits
    }






