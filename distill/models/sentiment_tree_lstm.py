import tensorflow as tf
from distill.layers.tree_lstm import TreeLSTM


class SentimentTreeLSTM(object):
  def __init__(self, config):
    self.config = config
    self.tree_lstm = TreeLSTM(input_dim=config.vocab_size,
                              hidden_dim=config.embed_size,
                              output_dim=config.label_size)


  def build_graph(self):
    self.tree_lstm.create_vars()


  def apply(self, examples):
    example_id, length, is_leaf, left_children, right_children, node_word_ids, _, labels = examples

    tree_lstm_output_dic = self.tree_lstm.apply(examples)

    logits = tree_lstm_output_dic['logits']
    root_logits = tree_lstm_output_dic['root_logits']
    root_indices = tree_lstm_output_dic['root_indices']

    root_prediction = tf.argmax(root_logits, axis=1)


    logits_mask = tf.cast(tf.sequence_mask(length), dtype=tf.float32)
    full_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=logits, labels=labels)
    masked_full_loss = full_loss * logits_mask
    full_loss = tf.reduce_sum(masked_full_loss)

    root_labels = tf.gather_nd(labels, root_indices)
    root_loss = tf.reduce_sum(
      tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=root_logits, labels=root_labels))


    root_accuracy = tf.reduce_mean(tf.cast(tf.math.equal(root_prediction, root_labels), dtype=tf.float32))

    return {'predictions': root_prediction,
            'logits': logits,
            'labels': labels,
            'root_logits': root_logits,
            'loss': root_loss,
            'full_loss': full_loss,
            'root_loss': root_loss,
            'root_accuracy': root_accuracy,
            'trainable_vars': tf.trainable_variables()}
