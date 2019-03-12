import tensorflow as tf
from distill.layers.lstm import LSTM
from distill.layers.bilstm import BiLSTM



class SentimentLSTM(object):
  def __init__(self, config, model=LSTM, scope="SentimentLSTM"):
    self.config = config
    self.scope=scope
    self.lstm = model(input_dim=config.input_dim,
                              hidden_dim=config.hidden_dim,
                              output_dim=config.output_dim,
                              input_keep_prob=config.input_dropout_keep_prob,
                              hidden_keep_prob=config.input_dropout_keep_prob,
                              attention_mechanism=self.config.attention_mechanism,
                              depth=config.depth,
                              sent_rep_mode=self.config.sent_rep_mode,
                              scope=scope)


  def build_graph(self, pretrained_word_embeddings):
    self.lstm.create_vars(pretrained_word_embeddings)

    # Output embedding
    self.output_embedding_mat = tf.get_variable("output_embedding_mat",
                                                [self.config.vocab_size, self.config.hidden_dim],
                                                dtype=tf.float32)

    self.output_embedding_bias = tf.get_variable("output_embedding_bias",
                                                 [self.config.vocab_size],
                                                 dtype=tf.float32)


  def apply(self, examples, is_train=True):
    example_ids, inputs, inputs_length, labels = examples
    inputs_mask = tf.sequence_mask(inputs_length)
    lstm_output_dic = self.lstm.apply(inputs=inputs, inputs_length=labels, is_train=is_train)

    seq_states = lstm_output_dic['raw_outputs']

    def output_embedding(current_output):
      return tf.add(
        tf.matmul(
          current_output,
          tf.transpose(self.output_embedding_mat)),
        self.output_embedding_bias)

    logits = tf.map_fn(output_embedding, seq_states)
    predictions = tf.arg_max(logits, axis=1)
    logits = tf.reshape(logits, [-1, self.config.vocab_size])
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=tf.reshape(labels, [-1]),
      logits=logits) * tf.cast(tf.reshape(inputs_mask, [-1]), tf.float32)

    loss = tf.reduce_sum(loss)

    return {'loss': loss,
            'predictions': predictions,
            'logits': logits}





    tf.logging.info("labels")
    tf.logging.info(labels)

    root_accuracy = tf.reduce_mean(tf.cast(tf.math.equal(predictions, tf.expand_dims(labels,axis=-1)), dtype=tf.float32))
    total_matchings = tf.reduce_sum(tf.cast(tf.math.equal(predictions, tf.expand_dims(labels,axis=-1)), dtype=tf.float32))

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
            'trainable_vars': tf.trainable_variables(scope=self.scope),
            'sents_reps': lstm_output_dic['sents_reps']}
