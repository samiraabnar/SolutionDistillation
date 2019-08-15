import tensorflow as tf

from distill.data_util.prep_ptb import PTB
from distill.layers.embedding import Embedding, EmbeddingSharedWeights
from distill.layers.lstm import LSTM
from distill.layers.bilstm import BiLSTM



class LmLSTM(object):
  def __init__(self, config, task, model=LSTM, scope="LMLSTM"):
    self.hparams = config
    self.scope=scope
    self.task = task
    self.eos_id = self.task.eos_id
    self.lstm = model(hidden_dim=config.hidden_dim,
                      output_dim=config.vocab_size,
                      hidden_keep_prob=config.input_dropout_keep_prob,
                      attention_mechanism=self.hparams.attention_mechanism,
                      depth=config.encoder_depth,
                      sent_rep_mode=self.hparams.sent_rep_mode,
                      scope=scope)


  def build_graph(self, reuse=tf.AUTO_REUSE):
    with tf.variable_scope(self.scope, reuse=reuse):
      self.embedding_layer = EmbeddingSharedWeights(vocab_size=self.hparams.vocab_size,
                                                          embedding_dim=self.hparams.embedding_dim,
                                                          pretrained_embeddings=pretrained_embeddings,
                                                          scope="InputEmbed")

      self.embedding_layer.create_vars(is_train=self.hparams.train_embeddings)
      if not self.task.share_input_output_embeddings:
        self.output_embedding_layer = EmbeddingSharedWeights(vocab_size=len(self.task.target_vocab),
                                                             embedding_dim=self.lstm_decoder.sent_rep_dim,
                                                             scope="OutputEmbed")
        self.output_embedding_layer.create_vars()
      else:
        self.output_embedding_layer = self.input_embedding_layer

      self.lstm.create_vars()

      # Output embedding
      self.output_embedding_mat = tf.get_variable("output_embedding_mat",
                                                  [self.hparams.vocab_size, self.hparams.hidden_dim],
                                                  dtype=tf.float32)

      self.output_embedding_bias = tf.get_variable("output_embedding_bias",
                                                   [self.hparams.vocab_size],
                                                   dtype=tf.float32)


  def apply(self, examples, is_train=True, reuse=tf.AUTO_REUSE, target_length=None):
    inputs, labels, inputs_length = examples
    tf.logging.info(inputs_length)
    inputs_mask = tf.sequence_mask(inputs_length)

    batch_size = tf.shape(inputs)[0]
    with tf.variable_scope(self.scope, reuse=reuse):
      embedded_input = self.embedding_layer.apply(inputs, is_train)
      lstm_output_dic = self.lstm.apply(inputs=embedded_input, inputs_length=inputs_length, is_train=is_train)

      seq_states = lstm_output_dic['raw_outputs']



      logits = self.output_embedding_layer.linear(seq_states)

      tf.logging.info("states")
      tf.logging.info(seq_states)
      tf.logging.info("logits")
      tf.logging.info(logits)
      predictions = tf.argmax(logits, axis=-1)

      flat_logits = tf.reshape(logits, [-1, tf.shape(logits)[-1]])
      tf.logging.info(flat_logits)

      flat_labels = tf.reshape(labels, [-1])
      tf.logging.info(flat_labels)

      flat_mask = tf.cast(tf.reshape(inputs_mask, [-1]), tf.float32)

      flat_predictions = tf.reshape(predictions, [-1])

      if is_train:
        loss = tf.nn.sampled_softmax_loss(
          weights=self.output_embedding_layer.shared_weights,
          labels=tf.reshape(labels, [-1, 1]),
          inputs=tf.reshape(seq_states, [-1, 128]),
          num_classes=self.hparams.vocab_size,
          num_sampled=1000,
          partition_strategy="div")
      else:
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=flat_labels,
          logits=flat_logits) * flat_mask

      perplexity = tf.reduce_mean(tf.exp(loss))
      loss = tf.reduce_mean(loss)

      tf.logging.info(flat_labels)
      tf.logging.info(flat_mask)
      accuracy = tf.reduce_sum(tf.cast(tf.equal(flat_labels, flat_predictions), dtype=tf.float32) * flat_mask) / tf.reduce_sum(flat_mask)
      correct_predictions = tf.cast(tf.equal(predictions, labels), dtype=tf.float32) * tf.cast(inputs_mask, dtype=tf.float32)
      correct_sequences = tf.reduce_min(correct_predictions, axis=-1)
      sequence_accuracy = tf.reduce_sum(correct_sequences) / tf.cast(batch_size, dtype=tf.float32)

    return {'inputs': inputs,
            'loss': loss,
            'predictions': predictions,
            'logits': logits,
            'accuracy': accuracy,
            'sequence_accuracy': sequence_accuracy,
            'perplexity': perplexity,
            'trainable_vars': tf.trainable_variables(scope=self.scope)}


if __name__ == '__main__':
  from distill.data_util.prep_algorithmic import AlgorithmicIdentityDecimal40, AlgorithmicIdentityBinary40
  import numpy as np

  tf.logging.set_verbosity(tf.logging.INFO)

  bin_iden = PTB('data/ptb')

  dataset = tf.data.TFRecordDataset(bin_iden.get_tfrecord_path(mode="train"))
  dataset = dataset.map(bin_iden.parse_examples)
  dataset = dataset.padded_batch(5, padded_shapes=bin_iden.get_padded_shapes())
  iterator = dataset.make_initializable_iterator()

  example = iterator.get_next()


  class Config(object):
    def __init__(self):
      self.vocab_size = bin_iden.vocab_length
      self.hidden_dim = 32
      self.output_dim = len(bin_iden.target_vocab),
      self.embedding_dim = 32
      self.input_dropout_keep_prob = 0.5
      self.hidden_dropout_keep_prob = 0.5
      self.attention_mechanism = None
      self.encoder_depth = 1
      self.decoder_depth = 1
      self.sent_rep_mode = "final"
      self.scope = "lstm_seq2seq"
      self.train_embeddings=False


  print("eos id: ", bin_iden.eos_id)
  model = LmLSTM(Config(), task=bin_iden, scope="LMLSTM")
  model.create_vars(reuse=False)

  input, target,_,_= example
  _ = model.apply(example, is_train=True, target_length=bin_iden.target_length)
  outputs = model.apply(example, is_train=False, target_length=bin_iden.target_length)

  predictions = outputs['predictions']

  global_step = tf.train.get_or_create_global_step()
  scaffold = tf.train.Scaffold(local_init_op=tf.group(tf.local_variables_initializer(),
                                                      iterator.initializer))

  accuracy = tf.equal(predictions, target)
  with tf.train.MonitoredTrainingSession(checkpoint_dir='logs/test_lm_lstm', scaffold=scaffold) as sess:
    for _ in np.arange(1):
      acc, _inputs, _targets, _predictions, _logits = sess.run([accuracy, input, target, predictions, outputs['logits']])

      print("input: ", _inputs)
      print("targets: ", _targets)

      print("predictions: ", _predictions)
      print("logits: ", _logits)

      print(acc)