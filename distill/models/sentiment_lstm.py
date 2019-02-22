import tensorflow as tf
from distill.layers.lstm import LSTM


class SentimentLSTM(object):
  def __init__(self, config, scope="SentimentLSTM"):
    self.config = config
    self.scope=scope
    self.lstm = LSTM(input_dim=config.input_dim,
                              hidden_dim=config.hidden_dim,
                              output_dim=config.output_dim,
                              input_keep_prob=config.input_dropout_keep_prob,
                              hidden_keep_prob=config.input_dropout_keep_prob,
                              depth=config.depth,
                              scope=scope)


  def build_graph(self, pretrained_word_embeddings):
    self.lstm.create_vars(pretrained_word_embeddings)


  def apply(self, examples):
    example_id, length, is_leaf, left_children, right_children, node_word_ids, labels, binary_labels, \
    root_label, root_binary_label, seq_lengths, seq_inputs = examples

    labels = root_binary_label

    lstm_output_dic = self.lstm.apply(inputs=seq_inputs, inputs_length=seq_lengths)

    logits = lstm_output_dic['logits']

    predictions = tf.argmax(logits, axis=1)

    loss = tf.reduce_sum(
      tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    root_accuracy = tf.reduce_mean(tf.cast(tf.math.equal(predictions, labels), dtype=tf.float32))
    total_matchings = tf.reduce_sum(tf.cast(tf.math.equal(predictions, labels), dtype=tf.float32))

    return {'predictions': predictions,
            'logits': logits,
            'labels': labels,
            'loss': loss,
            'root_loss': loss,
            'root_accuracy': root_accuracy,
            'raw_outputs': lstm_output_dic['raw_outputs'],
            'embedded_inputs': lstm_output_dic['embedded_inputs'],
            'raw_inputs': lstm_output_dic['raw_inputs'],
            'total_matchings': total_matchings,
            'trainable_vars': tf.trainable_variables(scope=self.scope)}
