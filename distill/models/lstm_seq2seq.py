import tensorflow as tf

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
                      depth=hparams.depth,
                      sent_rep_mode=self.hparams.sent_rep_mode,
                      scope=scope+"_encoder")
    self.lstm_decoder = model(hidden_dim=hparams.hidden_dim,
                              output_dim=hparams.hidden_dim,
                              hidden_keep_prob=hparams.hidden_dropout_keep_prob,
                              attention_mechanism=self.hparams.attention_mechanism,
                              depth=hparams.depth,
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
        elif target_length == 1:
          #This means the task is classification, we need neither teacher forcing nor the autoregressive process.
          transpose_embedded_targets = tf.transpose(embedded_targets, [1, 0, 2])
          decoder_inputs = tf.map_fn(compute_decoding_step_input,
                                     transpose_embedded_targets)  # (Length, batch_size, hidden_dim)
          decoder_inputs = tf.transpose(decoder_inputs, [1, 0, 2])
          tf.logging.info('decoder_inputs')
          tf.logging.info(decoder_inputs)
          lstm_decoder_output_dic = self.lstm_decoder.apply(inputs=decoder_inputs, inputs_length=targets_length,
                                                            is_train=is_train)
        else:
          #When generating a sequence,
          lstm_decoder_output_dic = self.lstm_decoder.predict(inputs_length=inputs_lengths,
                                                              target_length=target_length,
                                                              compute_decoding_step_input_fn=compute_decoding_step_input,
                                                              embedding_layer=self.output_embedding_layer,eos_id=self.eos_id, is_train=is_train)

      outputs = lstm_decoder_output_dic['seq_outputs']
      tf.logging.info("outputs")
      tf.logging.info(outputs)
      if is_train:
        tf.nn.dropout(
          outputs, self.hparams.hidden_dropout_keep_prob)

      output_mask = tf.cast(tf.sequence_mask(lstm_decoder_output_dic['outputs_lengths'], tf.shape(outputs)[1]), dtype=tf.int64)

      logits = self.output_embedding_layer.linear(outputs)
      predictions = tf.argmax(logits, axis=-1) * output_mask

    return {'logits': logits,
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

  bin_iden = AlgorithmicIdentityBinary40('data/alg')

  dataset = tf.data.TFRecordDataset(bin_iden.get_tfrecord_path(mode="train"))
  dataset = dataset.map(bin_iden.parse_examples)
  dataset = dataset.padded_batch(1, padded_shapes=bin_iden.get_padded_shapes())
  iterator = dataset.make_initializable_iterator()

  example = iterator.get_next()


  class Config(object):
    def __init__(self):
      self.vocab_size = bin_iden.vocab_length
      self.hidden_dim = 32
      self.output_dim = self.vocab_size
      self.embedding_dim = 10
      self.input_dropout_keep_prob = 0.5
      self.hidden_dropout_keep_prob = 0.5
      self.attention_mechanism = None
      self.depth = 1
      self.sent_rep_mode = "all"
      self.scope = "lstm_seq2seq"


  print("eos id: ", bin_iden.eos_id)
  model = BidiLSTMSeq2Seq(Config(), eos_id=bin_iden.eos_id, scope="Seq2SeqLSTM")
  model.create_vars(reuse=False)

  input, _,_,_= example
  _ = model.apply(example, is_train=True)
  outputs = model.apply(example, is_train=False)

  predictions = outputs['predictions']

  global_step = tf.train.get_or_create_global_step()
  scaffold = tf.train.Scaffold(local_init_op=tf.group(tf.local_variables_initializer(),
                                                      iterator.initializer))
  with tf.train.MonitoredTrainingSession(checkpoint_dir='logs', scaffold=scaffold) as sess:
    for _ in np.arange(10):
      print(sess.run([input, predictions]))