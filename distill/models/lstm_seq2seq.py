import tensorflow as tf

from distill.data_util.prep_arithmatic import Arithmatic
from distill.data_util.prep_sst import SST
from distill.layers.embedding import Embedding, EmbeddingSharedWeights
from distill.layers.lstm import LSTM
from distill.layers.bilstm import BiLSTM



class LSTMSeq2Seq(object):
  def __init__(self, hparams, task, model=LSTM, scope="Seq2SeqLSTM"):
    self.hparams = hparams
    self.scope=scope
    self.task = task
    self.eos_id = self.task.eos_id
    self.lstm_encoder = model(hidden_dim=hparams.hidden_dim,
                      output_dim=hparams.hidden_dim,
                      hidden_keep_prob=hparams.hidden_dropout_keep_prob,
                      attention_mechanism=self.hparams.attention_mechanism,
                      depth=hparams.encoder_depth,
                      sent_rep_mode=self.hparams.sent_rep_mode,
                      scope=scope+"_encoder")
    self.lstm_decoder = model(hidden_dim=hparams.hidden_dim,
                              output_dim=hparams.hidden_dim,
                              hidden_keep_prob=hparams.hidden_dropout_keep_prob,
                              attention_mechanism=self.hparams.attention_mechanism,
                              depth=hparams.decoder_depth,
                              sent_rep_mode=self.hparams.sent_rep_mode,
                              scope=scope+"_decoder")



  def create_vars(self, reuse=False, pretrained_embeddings=None):
    with tf.variable_scope(self.scope, reuse=reuse):
      self.input_embedding_layer = EmbeddingSharedWeights(vocab_size=self.hparams.vocab_size,
                                       embedding_dim=self.hparams.embedding_dim,
                                       pretrained_embeddings=pretrained_embeddings,
                                       scope="InputEmbed")

      self.input_embedding_layer.create_vars(is_train=self.hparams.train_embeddings)
      if not self.task.share_input_output_embeddings:
        self.output_embedding_layer = EmbeddingSharedWeights(vocab_size=len(self.task.target_vocab),
                                       embedding_dim=self.lstm_decoder.sent_rep_dim, scope="OutputEmbed")
        self.output_embedding_layer.create_vars()
      else:
        self.output_embedding_layer = self.input_embedding_layer

      with tf.variable_scope("encoder"):
        self.lstm_encoder.create_vars()
      with tf.variable_scope("decoder"):
        self.lstm_decoder.create_vars()



  def apply(self, examples, target_length=None, is_train=True, reuse=tf.AUTO_REUSE):

    inputs, targets, inputs_lengths, targets_length = examples
    tf.logging.info(inputs)

    with tf.variable_scope(self.scope, reuse=reuse):
      embedded_inputs = self.input_embedding_layer.apply(inputs)
      embedded_targets = self.output_embedding_layer.apply(targets)

      if is_train:
        embedded_inputs = tf.nn.dropout(
          embedded_inputs, keep_prob=self.hparams.input_dropout_keep_prob)

      with tf.variable_scope("encoder"):
        lstm_encoder_output_dic = self.lstm_encoder.apply(inputs=embedded_inputs, inputs_length=inputs_lengths, is_train=is_train)
        #encoder_output = tf.expand_dims(lstm_encoder_output_dic['sents_reps'],1)
        #encoder_output = tf.tile(encoder_output,[1,tf.shape(targets)[1],1])

        if is_train:
          encoder_outputs = lstm_encoder_output_dic['raw_outputs']
          encoder_outputs = tf.nn.dropout(encoder_outputs, self.hparams.hidden_dropout_keep_prob)

        if self.hparams.sent_rep_mode == "none":
          def compute_decoding_step_input(current_decoder_input):
            aggregiated_encoder_output = tf.reduce_mean(encoder_outputs, axis=1)
            if is_train:
              aggregiated_encoder_output = tf.nn.dropout(aggregiated_encoder_output,
                                                         self.hparams.hidden_dropout_keep_prob)
            return aggregiated_encoder_output
        else:
          def compute_decoding_step_input(current_decoder_input):
            aggregiated_encoder_output = lstm_encoder_output_dic['sents_reps']
            if is_train:
              aggregiated_encoder_output = tf.nn.dropout(aggregiated_encoder_output,
                                                         self.hparams.hidden_dropout_keep_prob)

            return aggregiated_encoder_output

      with tf.variable_scope("decoder"):
        if is_train and target_length is None:
          # Shift targets to the right, and remove the last element
          embedded_targets = tf.pad(
            embedded_targets, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
          transpose_embedded_targets = tf.transpose(embedded_targets, [1,0,2])
          decoder_inputs = tf.map_fn(compute_decoding_step_input, transpose_embedded_targets) #(Length, batch_size, hidden_dim)
          decoder_inputs = tf.transpose(decoder_inputs,[1,0,2])
          tf.logging.info('decoder_inputs')
          tf.logging.info(decoder_inputs)
          decoder_inputs = tf.concat([decoder_inputs, embedded_targets], axis=-1)
          tf.logging.info('decoder_inputs')
          tf.logging.info(decoder_inputs)
          lstm_decoder_output_dic = self.lstm_decoder.apply(inputs=decoder_inputs, inputs_length=targets_length,
                                                            is_train=is_train)
          outputs = lstm_decoder_output_dic['seq_outputs']
          outputs_lengths = lstm_decoder_output_dic['outputs_lengths']
          tf.logging.info("outputs")
          tf.logging.info(outputs)

        elif target_length == 1:
          # This means the task is classification.
          transpose_embedded_targets = tf.transpose(embedded_targets, [1, 0, 2])
          decoder_inputs = tf.map_fn(compute_decoding_step_input,
                                     transpose_embedded_targets)  # (Length, batch_size, hidden_dim)
          outputs = tf.transpose(decoder_inputs, [1, 0, 2])

        else:
          #When generating a sequence,
          lstm_decoder_output_dic = self.lstm_decoder.predict(inputs_length=inputs_lengths,
                                                              target_length=target_length,
                                                              compute_decoding_step_input_fn=compute_decoding_step_input,
                                                              embedding_layer=self.output_embedding_layer,eos_id=self.eos_id, is_train=is_train)
          outputs = lstm_decoder_output_dic['seq_outputs']
          outputs_lengths = lstm_decoder_output_dic['outputs_lengths']
          tf.logging.info("outputs")
          tf.logging.info(outputs)

      output_mask = tf.cast(tf.sequence_mask(outputs_lengths, tf.shape(outputs)[1]), dtype=tf.int64)



      logits = self.output_embedding_layer.linear(outputs)
      tf.logging.info("logits")
      tf.logging.info(logits)
      predictions = tf.cast(tf.argmax(logits, axis=-1) * output_mask, dtype=tf.int64)

    return {'logits': logits,
            'output_mask': output_mask,
            'outputs': outputs,
            'predictions': predictions,
            'targets': targets,
            'trainable_vars': tf.trainable_variables(scope=self.scope),
            }

  def _get_symbols_to_logits_fn(self):
    """Returns a decoding function that calculates logits of the next tokens."""

    def symbols_to_logits_fn(ids, cache):
      """Generate logits for next potential IDs.
      Args:
        ids: Current decoded sequences.
          int tensor with shape [batch_size * beam_size, i + 1]
        i: Loop index
        cache: dictionary of values storing the encoder output, encoder-decoder
          attention bias, and previous decoder attention values.
      Returns:
        Tuple of
          (logits with shape [batch_size * beam_size, vocab_size],
           updated cache values)
      """
      # Set decoder input to the last generated IDs
      decoder_input = ids[:, -1:]

      # Preprocess decoder input by getting embeddings and adding timing signal.
      decoder_input = self.embedding_layer.apply(decoder_input)

      decoder_outputs = self.lstm_decoder.apply(
        decoder_input, cache.get("encoder_outputs"), cache)

      logits = self.embedding_layer.linear(decoder_outputs)
      logits = tf.squeeze(logits, axis=[1])

      return logits, cache

    return symbols_to_logits_fn



class BidiLSTMSeq2Seq(LSTMSeq2Seq):
  def __init__(self, hparams, task, scope="Seq2SeqBiLSTM"):
    super(BidiLSTMSeq2Seq, self,).__init__(hparams, task, model=BiLSTM, scope=scope)


if __name__ == '__main__':
  from distill.data_util.prep_algorithmic import AlgorithmicIdentityDecimal40, AlgorithmicIdentityBinary40
  import numpy as np

  tf.logging.set_verbosity(tf.logging.INFO)

  bin_iden = Arithmatic('data/arithmatic') #SST(data_path="data/sst/",
             #    add_subtrees=True,
             #    pretrained=False)

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
      self.sent_rep_mode = "all"
      self.scope = "lstm_seq2seq"
      self.train_embeddings=False


  print("eos id: ", bin_iden.eos_id)
  model = LSTMSeq2Seq(Config(), task=bin_iden, scope="Seq2SeqLSTM")
  model.create_vars(reuse=False)

  input, target,_,_= example
  _ = model.apply(example, is_train=True, target_length=bin_iden.target_length)
  outputs = model.apply(example, is_train=False, target_length=bin_iden.target_length)

  predictions = outputs['predictions']

  global_step = tf.train.get_or_create_global_step()
  scaffold = tf.train.Scaffold(local_init_op=tf.group(tf.local_variables_initializer(),
                                                      iterator.initializer))

  accuracy = tf.equal(predictions, target)
  with tf.train.MonitoredTrainingSession(checkpoint_dir='logs/test_lstm_seq2seq', scaffold=scaffold) as sess:
    for _ in np.arange(1):
      acc, _inputs, _targets, _predictions, _logits = sess.run([accuracy, input, target, predictions, outputs['logits']])

      print("input: ", _inputs)
      print("targets: ", _targets)

      print("predictions: ", _predictions)
      print("logits: ", _logits)

      print(acc)
