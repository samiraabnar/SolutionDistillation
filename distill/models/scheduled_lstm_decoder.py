import tensorflow as tf
from distill.data_util.prep_ptb import PTB
from distill.layers.embedding import EmbeddingSharedWeights
from tensorflow.contrib.seq2seq.python.ops.basic_decoder import BasicDecoder, BasicDecoderOutput
from tensorflow.contrib.seq2seq.python.ops.helper import TrainingHelper, ScheduledEmbeddingTrainingHelper, ScheduledOutputTrainingHelper
from tensorflow.python.layers.core import Dense


class ScheduledLSTMDecoder(object):
  def __init__(self, config, task, model=None, scope="LSTMDecoder"):
    self.hparams = config
    self.scope = scope
    self.task = task
    self.eos_id = self.task.eos_id
    self.normalizer = tf.contrib.layers.layer_norm
    self.initializer = tf.contrib.layers.xavier_initializer()

  def create_vars(self, reuse=tf.AUTO_REUSE, pretrained_embeddings=None):
    with tf.variable_scope(self.scope, reuse=reuse):
      self.embedding_layer = EmbeddingSharedWeights(vocab_size=self.hparams.vocab_size,
                                                          embedding_dim=self.hparams.embedding_dim,
                                                          pretrained_embeddings=pretrained_embeddings,
                                                          scope="InputEmbed")
      self.embedding_layer.create_vars()
      self.output_embedding_layer = Dense(self.hparams.vocab_size)

      with tf.variable_scope("LSTM_Cells"):
        lstm0 = tf.nn.rnn_cell.LSTMCell(self.hparams.hidden_dim, forget_bias=1.0,
                                        initializer=self.initializer, name="L0")
        dropout_lstm0 = tf.contrib.rnn.DropoutWrapper(lstm0,
                                                      output_keep_prob=self.hparams.hidden_dropout_keep_prob,
                                                      variational_recurrent=True,
                                                      dtype=tf.float32)

        lstms = [lstm0]
        drop_lstms = [dropout_lstm0]

        lstm = tf.nn.rnn_cell.LSTMCell(self.hparams.hidden_dim, forget_bias=1.0,
                                       initializer=self.initializer, name="L1")
        dropout_lstm = tf.contrib.rnn.DropoutWrapper(lstm,
                                                      output_keep_prob=self.hparams.hidden_dropout_keep_prob,
                                                      variational_recurrent=True,
                                                      dtype=tf.float32
                                                     )
        if self.hparams.decoder_depth > 1:
          lstms.extend([lstm] * (self.hparams.decoder_depth-1))
          drop_lstms.extend([dropout_lstm] * (self.hparams.decoder_depth - 1))

        self.multi_lstm_cell = tf.contrib.rnn.MultiRNNCell(lstms)
        self.multi_dropout_lstm_cell = tf.contrib.rnn.MultiRNNCell(drop_lstms)

  def apply(self, examples, is_train=True, reuse=tf.AUTO_REUSE, target_length=None):
    inputs, targets, inputs_length, targets_lengths = examples
    inputs_length = tf.cast(inputs_length, dtype=tf.int32)
    targets_lengths = tf.cast(targets_lengths, dtype=tf.int32)

    tf.logging.info(inputs_length)
    inputs_mask = tf.sequence_mask(inputs_length)

    batch_size = tf.shape(inputs)[0]
    with tf.variable_scope(self.scope, reuse=reuse):
      embedded_inputs = self.embedding_layer.apply(inputs)
      rnn_cell = self.multi_lstm_cell
      if is_train:
        embedded_inputs = tf.nn.dropout(embedded_inputs, keep_prob=self.hparams.input_dropout_keep_prob)
        rnn_cell = self.multi_dropout_lstm_cell

      helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(
        embedded_inputs,
        targets_lengths,
        self.embedding_layer.shared_weights,
        sampling_probability=0.25)

      decoder = tf.contrib.seq2seq.BasicDecoder(
        rnn_cell,
        helper,
        rnn_cell.zero_state(batch_size, dtype=tf.float32),
        output_layer=self.output_embedding_layer)

      outputs, state, seq_len = tf.contrib.seq2seq.dynamic_decode(decoder)
      logits = outputs.rnn_output
      predictions = tf.argmax(logits, axis=-1)

      loss = tf.contrib.seq2seq.sequence_loss(
        logits,
        targets,
        weights=tf.cast(inputs_mask, dtype=tf.float32),
        average_across_timesteps=True,
        average_across_batch=True)

    return {'inputs': inputs,
            'loss': loss,
            'predictions': predictions,
            'logits': logits,
            'targets': targets,
            'trainable_vars': tf.trainable_variables(scope=self.scope)}

  def sample(self, number_of_sample=3, target_len=40):
    initial_constant = bin_iden.word2id[self.task.start_token]
    inputs = tf.ones((number_of_sample), dtype=tf.int32) * initial_constant
    inputs_length = tf.map_fn(lambda  x: 1, inputs)
    inputs_length = tf.cast(inputs_length, dtype=tf.int32)
    tf.logging.info(inputs_length)
    batch_size = tf.shape(inputs)[0]
    with tf.variable_scope(self.scope, reuse=True):
      embedded_inputs = self.embedding_layer.apply(inputs)
      rnn_cell = self.multi_lstm_cell

      # Inference
      inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
        self.embedding_layer.shared_weights,
        tf.fill([batch_size], self.task.word2id[self.task.start_token]),
        self.task.eos_id)

      # Inference Decoder
      inference_decoder = tf.contrib.seq2seq.BasicDecoder(
        rnn_cell, inference_helper,
        rnn_cell.zero_state(batch_size, dtype=tf.float32),
        output_layer=self.output_embedding_layer)

      # We should specify maximum_iterations, it can't stop otherwise.
      source_sequence_length = target_len
      maximum_iterations = tf.round(tf.reduce_max(source_sequence_length))

      # Dynamic decoding
      outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
        inference_decoder, maximum_iterations=maximum_iterations)

      predictions = outputs.sample_id

    return predictions

if __name__ == '__main__':
  import numpy as np

  tf.logging.set_verbosity(tf.logging.INFO)

  bin_iden = PTB('data/ptb')

  dataset = tf.data.TFRecordDataset(bin_iden.get_tfrecord_path(mode="train"))
  dataset = dataset.map(bin_iden.parse_examples)
  dataset = dataset.padded_batch(2, padded_shapes=bin_iden.get_padded_shapes())
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
  model = ScheduledLSTMDecoder(Config(), task=bin_iden, scope="LMLSTM")
  model.create_vars(reuse=False)

  input, target,_,_= example
  _ = model.apply(example, is_train=True, target_length=bin_iden.target_length)
  outputs = model.apply(example, is_train=False, target_length=bin_iden.target_length)


  samples = model.sample()
  predictions = outputs['predictions']

  global_step = tf.train.get_or_create_global_step()
  scaffold = tf.train.Scaffold(local_init_op=tf.group(tf.local_variables_initializer(),
                                                      iterator.initializer))

  accuracy = tf.equal(predictions, target)

  with tf.train.MonitoredTrainingSession(checkpoint_dir='logs/test_sss_lstm', scaffold=scaffold) as sess:
    for _ in np.arange(1):
      samples, acc, _inputs, _targets, _predictions, _logits = sess.run([samples ,accuracy, input, target, predictions, outputs['logits']])

      print("input: ", _inputs)
      print("targets: ", _targets)
      print("samples:", samples)