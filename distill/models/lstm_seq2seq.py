import tensorflow as tf

from distill.layers.embedding import Embedding, EmbeddingSharedWeights
from distill.layers.lstm import LSTM
from distill.layers.bilstm import BiLSTM



class LSTMSeq2Seq(object):
  def __init__(self, config, model=LSTM, scope="Seq2SeqLSTM"):
    self.config = config
    self.scope=scope
    self.lstm_encoder = model(hidden_dim=config.hidden_dim,
                      output_dim=config.hidden_dim,
                      hidden_keep_prob=config.hidden_dropout_keep_prob,
                      attention_mechanism=self.config.attention_mechanism,
                      depth=config.depth,
                      sent_rep_mode=self.config.sent_rep_mode,
                      scope=scope+"_encoder")
    self.lstm_decoder = model(hidden_dim=config.hidden_dim,
                              output_dim=config.hidden_dim,
                              hidden_keep_prob=config.hidden_dropout_keep_prob,
                              attention_mechanism=self.config.attention_mechanism,
                              depth=config.depth,
                              sent_rep_mode=self.config.sent_rep_mode,
                              scope=scope+"_decoder")


  def create_vars(self, reuse=False):
    with tf.variable_scope(self.scope, reuse=reuse):
      self.embedding_layer = EmbeddingSharedWeights(vocab_size=self.config.vocab_size,
                                       embedding_dim=self.config.embedding_dim)
      self.embedding_layer.create_vars()
      with tf.variable_scope("encoder"):
        self.lstm_encoder.create_vars()
      with tf.variable_scope("decoder"):
        self.lstm_decoder.create_vars()

      # Output embedding
      self.output_embedding_mat = tf.get_variable("output_embedding_mat",
                                                  [self.config.embedding_dim, self.config.hidden_dim],
                                                  dtype=tf.float32)
      self.output_embedding_bias = tf.get_variable("output_embedding_bias",
                                                   [self.config.embedding_dim],
                                                   dtype=tf.float32)


  def apply(self, inputs, targets,input_lengths, target_length, is_train=True, reuse=tf.AUTO_REUSE):

    with tf.variable_scope(self.scope, reuse=reuse):
      embedded_inputs = self.embedding_layer.apply(inputs)
      embedded_targets = self.embedding_layer.apply(targets)
      with tf.variable_scope("encoder"):
        lstm_encoder_output_dic = self.lstm_encoder.apply(inputs=embedded_inputs, inputs_length=input_lengths, is_train=is_train)
        encoder_output = tf.expand_dims(lstm_encoder_output_dic['sents_reps'],1)
        encoder_output = tf.tile(encoder_output,[1,tf.shape(targets)[1],1])
        tf.logging.info("encoder_output")
        tf.logging.info(encoder_output)
      with tf.variable_scope("decoder"):
        decoder_inputs = tf.concat([encoder_output, embedded_targets], axis=-1)
        tf.logging.info('decoder_inputs')
        tf.logging.info(decoder_inputs)
        lstm_decoder_output_dic = self.lstm_decoder.apply(inputs=decoder_inputs, inputs_length=target_length, is_train=is_train)

      outputs = lstm_decoder_output_dic['seq_outputs']
      def output_embedding(current_output):
        return tf.add(
          tf.matmul(current_output, tf.transpose(self.output_embedding_mat)),
          self.output_embedding_bias)

      outputs = tf.map_fn(output_embedding, outputs)

      logits = self.embedding_layer.linear(outputs)
      predictions = tf.argmax(logits, axis=-1)

    return {'logits': logits,
            'outputs': outputs,
            'predictions': predictions,
            'targets': targets,
            'trainable_vars': tf.trainable_variables(scope=self.scope),
            }


class BidiLSTMSeq2Seq(object):
  def __init__(self, config, scope="Seq2SeqLSTM"):
    super(BidiLSTMSeq2Seq, self,).__init__(config, model=BiLSTM, scope=scope)


if __name__ == '__main__':
  from distill.data_util.prep_algorithmic import AlgorithmicIdentityBinary40

  tf.logging.set_verbosity(tf.logging.INFO)

  bin_iden = AlgorithmicIdentityBinary40('data/alg')

  dataset = tf.data.TFRecordDataset(bin_iden.get_tfrecord_path(mode="train"))
  dataset = dataset.map(bin_iden.parse_examples)
  dataset = dataset.padded_batch(1, padded_shapes=bin_iden.get_padded_shapes())
  iterator = dataset.make_initializable_iterator()

  example = iterator.get_next()
  inputs, targets, inputs_lengths, targets_lengths = example


  class Config(object):
    def __init__(self):
      self.vocab_size = bin_iden.num_symbols + 1
      self.hidden_dim = 32
      self.output_dim = self.vocab_size
      self.embedding_dim = 100
      self.input_dropout_keep_prob = 0.5
      self.hidden_dropout_keep_prob = 0.5
      self.attention_mechanism = None
      self.depth = 1
      self.sent_rep_mode = "all"
      self.scope = "lstm_seq2seq"



  model = LSTMSeq2Seq(Config(), model=LSTM, scope="Seq2SeqLSTM")
  model.create_vars(reuse=False)

  outputs = model.apply(inputs, targets, inputs_lengths, targets_lengths)
  predictions = outputs['predictions']

  global_step = tf.train.get_or_create_global_step()
  scaffold = tf.train.Scaffold(local_init_op=tf.group(tf.local_variables_initializer(),
                                                      iterator.initializer))
  with tf.train.MonitoredTrainingSession(checkpoint_dir='logs', scaffold=scaffold) as sess:
    print(sess.run([inputs, predictions]))