import tensorflow as tf


class BinaryTreeLSTMCell(tf.contrib.rnn.BasicLSTMCell):
  """LSTM with two state inputs.

  This is the model described in section 3.2 of 'Improved Semantic
  Representations From Tree-Structured Long Short-Term Memory
  Networks' <http://arxiv.org/pdf/1503.00075.pdf>, with recurrent
  dropout as described in 'Recurrent Dropout without Memory Loss'
  <http://arxiv.org/pdf/1603.05118.pdf>.
  """

  def __init__(self, num_units, keep_prob=1.0):
    """Initialize the cell.

    Args:
      num_units: int, The number of units in the LSTM cell.
      keep_prob: Keep probability for recurrent dropout.
    """
    super(BinaryTreeLSTMCell, self).__init__(num_units)
    self._keep_prob = keep_prob

  def __call__(self, inputs, state, scope=None):
    with tf.variable_scope(scope or type(self).__name__):
      lhs, rhs = state
      c0, h0 = lhs
      c1, h1 = rhs
      concat = tf.contrib.layers.linear(
          tf.concat([inputs, h0, h1], 1), 5 * self._num_units)

      # i = input_gate, j = new_input, f = forget_gate, o = output_gate
      i, j, f0, f1, o = tf.split(value=concat, num_or_size_splits=5, axis=1)

      j = self._activation(j)
      if not isinstance(self._keep_prob, float) or self._keep_prob < 1:
        j = tf.nn.dropout(j, self._keep_prob)

      new_c = (c0 * tf.sigmoid(f0 + self._forget_bias) +
               c1 * tf.sigmoid(f1 + self._forget_bias) +
               tf.sigmoid(i) * j)
      new_h = self._activation(new_c) * tf.sigmoid(o)

      new_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)

      return new_h, new_state