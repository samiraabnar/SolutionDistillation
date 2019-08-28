import tensorflow as tf

from distill.data_util.prep_ptb import PTB
from distill.layers.embedding import EmbeddingSharedWeights
from distill.layers.lstm import LSTM

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


  def create_vars(self, reuse=tf.AUTO_REUSE, pretrained_embeddings=None):
    with tf.variable_scope(self.scope, reuse=reuse):
      self.input_embedding_layer = EmbeddingSharedWeights(vocab_size=self.hparams.vocab_size,
                                                          embedding_dim=self.hparams.embedding_dim,
                                                          pretrained_embeddings=pretrained_embeddings,
                                                          scope="InputEmbed")

      self.input_embedding_layer.create_vars(is_train=self.hparams.train_embeddings)
      if not self.task.share_input_output_embeddings:
        self.output_embedding_layer = EmbeddingSharedWeights(vocab_size=len(self.task.target_vocab),
                                                             embedding_dim=self.hparams.hidden_dim,
                                                             scope="OutputEmbed")
        self.output_embedding_layer.create_vars()
      else:
        self.output_embedding_layer = self.input_embedding_layer

      self.output_bias = tf.get_variable(name="output_bias", initializer=tf.zeros((self.hparams.vocab_size)))
      self.lstm.create_vars(share_in_depth=False)


  def apply(self, examples, is_train=True, reuse=tf.AUTO_REUSE, target_length=None):
    inputs, targets, inputs_length, targets_lengths = examples
    tf.logging.info(inputs_length)
    inputs_mask = tf.sequence_mask(inputs_length)

    with tf.variable_scope(self.scope, reuse=reuse):
      embedded_inputs = self.input_embedding_layer.apply(inputs)
      if is_train:
        embedded_inputs = tf.nn.dropout(embedded_inputs, keep_prob=self.hparams.input_dropout_keep_prob)

      lstm_output_dic = self.lstm.apply(inputs=embedded_inputs, inputs_length=inputs_length, is_train=is_train)

      seq_states = lstm_output_dic['raw_outputs']
      logits = self.output_embedding_layer.linear(seq_states) + self.output_bias
      predictions = tf.argmax(logits, axis=-1)

      loss = tf.contrib.seq2seq.sequence_loss(
        logits,
        targets,
        weights=tf.cast(inputs_mask, dtype=tf.float32),
        average_across_timesteps=True,
        average_across_batch=True)

    return {'inputs': inputs,
            'loss': loss,
            'outputs': lstm_output_dic['raw_outputs'],
            'predictions': predictions,
            'logits': logits,
            'targets': targets,
            'trainable_vars': tf.trainable_variables(scope=self.scope)}

  def sample(self, inputs, inputs_length):

    with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
      if inputs is not None:
          embedded_inputs = self.input_embedding_layer.apply(inputs)
      else:
          embedded_inputs = None
          
      def compute_decoding_step_input(current_input):
        return None

      lstm_decoder_output_dic = self.lstm.predict(inputs_length=inputs_length,
                                                          target_length=40,
                                                          compute_decoding_step_input_fn=compute_decoding_step_input,
                                                          input_embedding_layer=self.input_embedding_layer,
                                                          output_embedding_layer=self.output_embedding_layer, eos_id=self.eos_id,
                                                          is_train=False,
                                                          initial_inputs=embedded_inputs)
      outputs = lstm_decoder_output_dic['seq_outputs']
      outputs_lengths = lstm_decoder_output_dic['outputs_lengths']
      output_mask = tf.cast(tf.sequence_mask(outputs_lengths, tf.shape(outputs)[1]), dtype=tf.int64)
      logits = self.output_embedding_layer.linear(outputs) + self.output_bias

      predictions = tf.cast(tf.argmax(logits, axis=-1) * output_mask, dtype=tf.int64)

      return predictions




if __name__ == '__main__':
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

  number_of_sample = 3
  initial_constant = bin_iden.word2id[bin_iden.start_token]
  sampling_initial_inputs = tf.ones((number_of_sample), dtype=tf.int32) * initial_constant
  tf.logging.info(sampling_initial_inputs)
  samples = model.sample(sampling_initial_inputs, tf.map_fn(lambda  x: 1, sampling_initial_inputs))
  with tf.train.MonitoredTrainingSession(checkpoint_dir='logs/test_lm_lstm', scaffold=scaffold) as sess:
    for _ in np.arange(1):
      acc, _inputs, _targets, _predictions, _logits = sess.run([accuracy, input, target, predictions, outputs['logits']])

      print("input: ", _inputs)
      print("targets: ", _targets)

      print("predictions: ", _predictions)
      print("logits: ", _logits)

      print(acc)

      samples = sess.run([samples])
      print(samples[0])
      for s in samples[0]:
        print(s)
        print(bin_iden.decode(s,bin_iden.id2word))
