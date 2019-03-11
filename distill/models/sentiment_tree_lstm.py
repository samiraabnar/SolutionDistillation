import tensorflow as tf
from distill.layers.tree_lstm import TreeLSTM


class SentimentTreeLSTM(object):
  def __init__(self, config, model=None, scope="SentimentTreeLSTM"):
    self.config = config
    self.scope = scope
    self.tree_lstm = TreeLSTM(input_dim=config.input_dim,
                              embedding_dim=config.embedding_dim,
                              hidden_dim=config.hidden_dim,
                              output_dim=config.output_dim,
                              input_keep_prob=config.input_dropout_keep_prob,
                              hidden_keep_prob=config.input_dropout_keep_prob,
                              depth=config.depth,
                              scope=self.scope
                              )



  def build_graph(self, pretrained_word_embeddings):
    self.tree_lstm.create_vars(pretrained_word_embeddings)


  def apply(self, examples, is_train=True):
    example_id, length, is_leaf, left_children, right_children, node_word_ids, labels, binary_labels, \
    root_label, root_binary_label, seq_lengths, seq_inputs = examples

    tree_lstm_output_dic = self.tree_lstm.apply(examples, is_train)

    logits = tree_lstm_output_dic['logits']
    root_logits = tree_lstm_output_dic['root_logits']
    root_indices = tree_lstm_output_dic['root_indices']


    if self.config.output_dim > 1:
      root_prediction = tf.argmax(root_logits, axis=-1)
    else:
      root_prediction = tf.cast(tf.round(tf.nn.sigmoid(root_logits)), tf.int64)

    tf.logging.info("root_prediction")
    tf.logging.info(root_prediction)

    if self.config.output_dim > 1:
      full_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=binary_labels)
      root_loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=root_logits, labels=root_binary_label))
    else:
      full_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                labels=tf.cast(tf.expand_dims(binary_labels, axis=-1), dtype=tf.float32)))
      root_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
          logits=root_logits, labels=tf.cast(tf.expand_dims(root_binary_label, axis=-1), dtype=tf.float32)))



    logits_mask = tf.cast(tf.sequence_mask(length), dtype=tf.float32)
    masked_full_loss = full_loss * logits_mask
    full_loss = tf.reduce_sum(masked_full_loss)






    root_accuracy = tf.reduce_mean(tf.cast(tf.math.equal(root_prediction, root_binary_label), dtype=tf.float32))

    return {'predictions': root_prediction,
            'logits': root_logits,
            'labels': binary_labels,
            'root_logits': root_logits,
            'loss': root_loss,
            'full_loss': full_loss,
            'root_loss': root_loss,
            'root_accuracy': root_accuracy,
            'trainable_vars': tf.trainable_variables(scope=self.scope),
            'sents_reps': tree_lstm_output_dic['sents_reps']}
