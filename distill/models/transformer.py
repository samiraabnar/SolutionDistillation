import tensorflow as tf
import numpy as np
from tensor2tensor.utils.beam_search import EOS_ID

from distill.common.beam_search import sequence_beam_search
from distill.common.layer_utils import get_decoder_self_attention_bias, get_position_encoding, get_padding_bias, get_padding
from distill.layers.attention import MultiHeadScaledDotProductAttention
from distill.layers.embedding import EmbeddingSharedWeights
from distill.layers.ffn_layer import FeedFowardNetwork
from distill.layers.pre_post_wrapper import PrePostProcessingWrapper, LayerNormalization

tpu = False

class TransformerEncoder(object):
  def __init__(self, hidden_dim, number_of_heads, depth, ff_filter_size, dropout_keep_prob, scope="TransformerEncoder"):
    self.hidden_dim = hidden_dim
    self.number_of_heads = number_of_heads
    self.depth = depth
    self.dropout_keep_prob = dropout_keep_prob
    self.ff_filter_size = ff_filter_size

    self.scope = scope

  def create_vars(self, reuse=False):

    with tf.variable_scope(self.scope, reuse=reuse):
      self.layers = []
      for i in np.arange(self.depth):
        # Create sublayers for each layer.
        self_attention_layer = MultiHeadScaledDotProductAttention(hidden_dim=self.hidden_dim,
                                                                  num_heads=self.number_of_heads,
                                                                  attention_dropout_keepprob=self.dropout_keep_prob,
                                                                  scope="Attention"+str(i))
        feed_forward_network = FeedFowardNetwork(hidden_size=self.hidden_dim,
                                                 filter_size=self.ff_filter_size,
                                                 relu_dropout_keepprob=self.dropout_keep_prob,
                                                 allow_pad=True,
                                                 scope="FF"+str(i))

        wrapped_self_attention = PrePostProcessingWrapper(layer=self_attention_layer, hidden_dim=self.hidden_dim,
                                   postprocess_dropout_keepprob = self.dropout_keep_prob)
        wrapped_self_attention.create_vars()
        wrapped_ff = PrePostProcessingWrapper(layer=feed_forward_network, hidden_dim=self.hidden_dim,
                                   postprocess_dropout_keepprob = self.dropout_keep_prob)
        wrapped_ff.create_vars()

        self.layers.append([wrapped_self_attention, wrapped_ff])

      # Create final layer normalization layer.
      self.output_normalization = LayerNormalization(self.hidden_dim)
      self.output_normalization.create_vars()

  def apply(self, inputs, attention_bias, inputs_padding, is_train=True,  reuse=tf.AUTO_REUSE):

    encoder_inputs = inputs
    with tf.variable_scope(self.scope, reuse=reuse):
      for n, layer in enumerate(self.layers):
        # Run inputs through the sublayers.
        self_attention_layer = layer[0]
        feed_forward_network = layer[1]

        tf.logging.info("encoder inputs:")
        tf.logging.info(encoder_inputs)

        encoder_inputs = self_attention_layer.apply(x=encoder_inputs, y=encoder_inputs, is_train=is_train, bias=attention_bias)
        encoder_inputs = feed_forward_network.apply(x=encoder_inputs, is_train=is_train,
                                                    padding=inputs_padding)

    return self.output_normalization.apply(encoder_inputs, is_train)


class TransformerDecoder(object):
  def __init__(self, hidden_dim, number_of_heads, depth, ff_filter_size, dropout_keep_prob, scope="TransformerDecoder"):
    self.hidden_dim = hidden_dim
    self.number_of_heads = number_of_heads
    self.depth = depth
    self.dropout_keep_prob = dropout_keep_prob
    self.ff_filter_size = ff_filter_size

    self.scope = scope

  def create_vars(self, reuse=False):

    with tf.variable_scope(self.scope, reuse=reuse):
      self.layers = []
      for i in np.arange(self.depth):
        # Create sublayers for each layer.
        self_attention_layer = MultiHeadScaledDotProductAttention(hidden_dim=self.hidden_dim,
                                                                  num_heads=self.number_of_heads,
                                                                  attention_dropout_keepprob=self.dropout_keep_prob,
                                                                  scope="SelfAttention"+str(i))
        enc_dec_attention_layer = MultiHeadScaledDotProductAttention(hidden_dim=self.hidden_dim,
                                                                  num_heads=self.number_of_heads,
                                                                  attention_dropout_keepprob=self.dropout_keep_prob,
                                                                  scope="EncDecAttention" + str(i))
        feed_forward_network = FeedFowardNetwork(self.hidden_dim,
                                                 self.ff_filter_size,
                                                 self.dropout_keep_prob,
                                                 allow_pad=True,
                                                 scope="FF"+str(i))

        wrapped_self_attention = PrePostProcessingWrapper(layer=self_attention_layer, hidden_dim=self.hidden_dim,
                                   postprocess_dropout_keepprob = self.dropout_keep_prob)
        wrapped_enc_dec_attention = PrePostProcessingWrapper(layer=enc_dec_attention_layer, hidden_dim=self.hidden_dim,
                                   postprocess_dropout_keepprob=self.dropout_keep_prob)
        wrapped_ff = PrePostProcessingWrapper(layer=feed_forward_network, hidden_dim=self.hidden_dim,
                                   postprocess_dropout_keepprob = self.dropout_keep_prob)

        wrapped_self_attention.create_vars()
        wrapped_enc_dec_attention.create_vars()
        wrapped_ff.create_vars()

        self.layers.append([wrapped_self_attention,wrapped_enc_dec_attention,wrapped_ff])

      # Create final layer normalization layer.
      self.output_normalization = LayerNormalization(self.hidden_dim)
      self.output_normalization.create_vars()

  def apply(self, inputs, encoder_outputs, decoder_self_attention_bias, attention_bias, cache=None, is_train=True,  reuse=tf.AUTO_REUSE):
    decoder_inputs = inputs
    with tf.variable_scope(self.scope, reuse=reuse):
      for n, layer in enumerate(self.layers):
        layer_name = "layer_%d" % n
        layer_cache = cache[layer_name] if cache is not None else None

        # Run inputs through the sublayers.
        self_attention_layer = layer[0]
        enc_dec_attention = layer[1]
        feed_forward_network = layer[2]

        decoder_inputs = self_attention_layer.apply(x=decoder_inputs, y=decoder_inputs, is_train=is_train,
                                                    bias=decoder_self_attention_bias, cache=layer_cache)
        decoder_inputs = self_attention_layer.apply(x=decoder_inputs, y=encoder_outputs, is_train=is_train,
                                                    bias=attention_bias)
        decoder_inputs = feed_forward_network.apply(x=decoder_inputs, is_train=is_train)

    return self.output_normalization.apply(decoder_inputs, is_train)


class UniversalTransformerEncoder(TransformerEncoder):
  def __init__(self, hidden_dim, number_of_heads, depth, ff_filter_size, dropout_keep_prob, scope="TransformerEncoder"):
    super(UniversalTransformerEncoder, self).__init__(hidden_dim, number_of_heads, depth, ff_filter_size, dropout_keep_prob, scope)

  def create_vars(self, reuse=False):

    with tf.variable_scope(self.scope, reuse=reuse):
      self.layers = []
      for i in np.arange(self.depth):
        # Create sublayers for each layer.
        self_attention_layer = MultiHeadScaledDotProductAttention(hidden_dim=self.hidden_dim,
                                                                  num_heads=self.number_of_heads,
                                                                  attention_dropout_keepprob=self.dropout_keep_prob,
                                                                  scope="Attention")
        feed_forward_network = FeedFowardNetwork(hidden_size=self.hidden_dim,
                                                 filter_size=self.ff_filter_size,
                                                 relu_dropout_keepprob=self.dropout_keep_prob,
                                                 allow_pad=True,
                                                 scope="FF")

        wrapped_self_attention = PrePostProcessingWrapper(layer=self_attention_layer, hidden_dim=self.hidden_dim,
                                   postprocess_dropout_keepprob = self.dropout_keep_prob)
        wrapped_self_attention.create_vars(reuse=tf.AUTO_REUSE)
        wrapped_ff = PrePostProcessingWrapper(layer=feed_forward_network, hidden_dim=self.hidden_dim,
                                   postprocess_dropout_keepprob = self.dropout_keep_prob)
        wrapped_ff.create_vars(reuse=tf.AUTO_REUSE)

        self.layers.append([wrapped_self_attention, wrapped_ff])

      # Create final layer normalization layer.
      self.output_normalization = LayerNormalization(self.hidden_dim)
      self.output_normalization.create_vars()


class UniversalTransformerDecoder(TransformerDecoder):
  def __init__(self, hidden_dim, number_of_heads, depth, ff_filter_size, dropout_keep_prob, scope="TransformerDecoder"):
    super(UniversalTransformerDecoder, self).__init__(hidden_dim, number_of_heads, depth, ff_filter_size, dropout_keep_prob, scope)


  def create_vars(self, reuse=False):

    with tf.variable_scope(self.scope, reuse=reuse):
      self.layers = []
      for i in np.arange(self.depth):
        # Create sublayers for each layer.
        self_attention_layer = MultiHeadScaledDotProductAttention(hidden_dim=self.hidden_dim,
                                                                  num_heads=self.number_of_heads,
                                                                  attention_dropout_keepprob=self.dropout_keep_prob,
                                                                  scope="SelfAttention")
        enc_dec_attention_layer = MultiHeadScaledDotProductAttention(hidden_dim=self.hidden_dim,
                                                                  num_heads=self.number_of_heads,
                                                                  attention_dropout_keepprob=self.dropout_keep_prob,
                                                                  scope="EncDecAttention")
        feed_forward_network = FeedFowardNetwork(self.hidden_dim,
                                                 self.ff_filter_size,
                                                 self.dropout_keep_prob,
                                                 allow_pad=True,
                                                 scope="FF")

        wrapped_self_attention = PrePostProcessingWrapper(layer=self_attention_layer, hidden_dim=self.hidden_dim,
                                   postprocess_dropout_keepprob = self.dropout_keep_prob)
        wrapped_enc_dec_attention = PrePostProcessingWrapper(layer=enc_dec_attention_layer, hidden_dim=self.hidden_dim,
                                   postprocess_dropout_keepprob=self.dropout_keep_prob)
        wrapped_ff = PrePostProcessingWrapper(layer=feed_forward_network, hidden_dim=self.hidden_dim,
                                   postprocess_dropout_keepprob = self.dropout_keep_prob)

        wrapped_self_attention.create_vars(reuse=tf.AUTO_REUSE)
        wrapped_enc_dec_attention.create_vars(reuse=tf.AUTO_REUSE)
        wrapped_ff.create_vars(reuse=tf.AUTO_REUSE)

        self.layers.append([wrapped_self_attention,wrapped_enc_dec_attention,wrapped_ff])

      # Create final layer normalization layer.
      self.output_normalization = LayerNormalization(self.hidden_dim)
      self.output_normalization.create_vars()




class Transformer(object):
  """Transformer model for sequence to sequence data.
  Implemented as described in: https://arxiv.org/pdf/1706.03762.pdf
  The Transformer model consists of an encoder and decoder. The input is an int
  sequence (or a batch of sequences). The encoder produces a continous
  representation, and the decoder uses the encoder output to generate
  probabilities for the output sequence.
  """

  def __init__(self, hparams, scope="Transformer"):
    self.hparams = hparams
    self.vocab_size = hparams.vocab_size
    self.hidden_dim = hparams.hidden_dim
    self.number_of_heads = hparams.number_of_heads
    self.depth = hparams.depth
    self.ff_filter_size = hparams.ff_filter_size
    self.dropout_keep_prob = hparams.hidden_dropout_keep_prob
    self.initializer_gain = hparams.initializer_gain
    self.scope = scope

  def create_vars(self, reuse=False):
    self.initializer = tf.variance_scaling_initializer(
      self.initializer_gain, mode="fan_avg", distribution="uniform")

    with tf.variable_scope(self.scope, initializer=self.initializer, reuse=tf.AUTO_REUSE):
      self.embedding_softmax_layer = EmbeddingSharedWeights(vocab_size=self.vocab_size, embedding_dim=self.hidden_dim,
                                                       method="matmul" if tpu else "gather")

      self.encoder_stack = TransformerEncoder(self.hidden_dim, self.number_of_heads, self.depth, self.ff_filter_size,
                                              self.dropout_keep_prob,
                                              scope="TransformerEncoder")
      self.decoder_stack = TransformerDecoder(self.hidden_dim, self.number_of_heads, self.depth, self.ff_filter_size,
                                              self.dropout_keep_prob,
                                              scope="TransformerDecoder")

      self.embedding_softmax_layer.create_vars()
      self.encoder_stack.create_vars(reuse=False)
      self.decoder_stack.create_vars(reuse=False)

  def apply(self, examples, reuse=tf.AUTO_REUSE, is_train=True):
    """Calculate target logits or inferred target sequences.
    Args:
      inputs: int tensor with shape [batch_size, input_length].
      targets: None or int tensor with shape [batch_size, target_length].
    Returns:
      If targets is defined, then return logits for each word in the target
      sequence. float tensor with shape [batch_size, target_length, vocab_size]
      If target is none, then generate output sequence one token at a time.
        returns a dictionary {
          output: [batch_size, decoded length]
          score: [batch_size, float]}
    """
    inputs, targets, inputs_lengths, targets_lengths = examples

    with tf.variable_scope(self.scope, initializer=self.initializer, reuse=tf.AUTO_REUSE):
      # Calculate attention bias for encoder self-attention and decoder
      # multi-headed attention layers.
      attention_bias = get_padding_bias(inputs)

      # Run the inputs through the encoder layer to map the symbol
      # representations to continuous representations.
      encoder_outputs = self.encode(inputs, attention_bias, is_train)
      tf.logging.info('encoder outputs')
      tf.logging.info(encoder_outputs)

      if targets is None or not is_train:
        output_dic = self.predict(encoder_outputs, attention_bias)
        predictions = output_dic['outputs']
        logits = tf.one_hot(indices=predictions, depth=self.hparams.vocab_size)
        tf.logging.info('predict logits')
        tf.logging.info(logits)
        outputs = None
      else:
        outputs = self.decode(targets, encoder_outputs, attention_bias, is_train)
        logits = self.embedding_softmax_layer.linear(outputs)

      predictions = tf.argmax(logits, axis=-1)

      return {'logits': logits,
              'outputs': outputs,
              'predictions': predictions,
              'targets': targets,
              'trainable_vars': tf.trainable_variables(scope=self.scope),
              }

  def encode(self, inputs, attention_bias, is_train=True):
    """Generate continuous representation for inputs.
    Args:
      inputs: int tensor with shape [batch_size, input_length].
      attention_bias: float tensor with shape [batch_size, 1, 1, input_length]
    Returns:
      float tensor with shape [batch_size, input_length, hidden_size]
    """
    with tf.variable_scope("encode", reuse=tf.AUTO_REUSE):
      embedded_inputs = self.embedding_softmax_layer.apply(inputs)
      inputs_padding = get_padding(inputs)

      with tf.name_scope("add_pos_encoding"):
        length = tf.shape(embedded_inputs)[1]
        pos_encoding = get_position_encoding(
            length, self.hidden_dim)
        encoder_inputs = embedded_inputs + pos_encoding

      if is_train:
        encoder_inputs = tf.nn.dropout(
            encoder_inputs, self.dropout_keep_prob)

      return self.encoder_stack.apply(encoder_inputs, attention_bias, inputs_padding, is_train)

  def decode(self, targets, encoder_outputs, attention_bias, is_train=True):
    """Generate logits for each value in the target sequence.
    Args:
      targets: target values for the output sequence.
        int tensor with shape [batch_size, target_length]
      encoder_outputs: continuous representation of input sequence.
        float tensor with shape [batch_size, input_length, hidden_size]
      attention_bias: float tensor with shape [batch_size, 1, 1, input_length]
    Returns:
      float32 tensor with shape [batch_size, target_length, vocab_size]
    """
    with tf.name_scope("decode"):
      # Prepare inputs to decoder layers by shifting targets, adding positional
      # encoding and applying dropout.
      decoder_inputs = self.embedding_softmax_layer.apply(targets)
      with tf.name_scope("shift_targets"):
        # Shift targets to the right, and remove the last element
        decoder_inputs = tf.pad(
          decoder_inputs, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
      with tf.name_scope("add_pos_encoding"):
        length = tf.shape(decoder_inputs)[1]
        decoder_inputs += get_position_encoding(
            length, self.hidden_dim)
      if is_train:
        decoder_inputs = tf.nn.dropout(
            decoder_inputs, self.dropout_keep_prob)

      # Run values
      decoder_self_attention_bias = get_decoder_self_attention_bias(
          length)


      outputs = self.decoder_stack.apply(
            decoder_inputs, encoder_outputs, decoder_self_attention_bias,
            attention_bias)


      return outputs

  def _get_symbols_to_logits_fn(self, max_decode_length):
    """Returns a decoding function that calculates logits of the next tokens."""

    timing_signal = get_position_encoding(
      max_decode_length + 1, self.hidden_dim)
    decoder_self_attention_bias = get_decoder_self_attention_bias(
      max_decode_length)

    def symbols_to_logits_fn(ids, i, cache):
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
      decoder_input = self.embedding_softmax_layer.apply(decoder_input)
      decoder_input += timing_signal[i:i + 1]

      self_attention_bias = decoder_self_attention_bias[:, :, i:i + 1, :i + 1]

      decoder_outputs = self.decoder_stack.apply(
        decoder_input, cache.get("encoder_outputs"), self_attention_bias,
        cache.get("encoder_decoder_attention_bias"), cache)
      logits = self.embedding_softmax_layer.linear(decoder_outputs)
      logits = tf.squeeze(logits, axis=[1])
      return logits, cache

    return symbols_to_logits_fn

  def predict(self, encoder_outputs, encoder_decoder_attention_bias):
    """Return predicted sequence."""
    batch_size = tf.shape(encoder_outputs)[0]
    input_length = tf.shape(encoder_outputs)[1]
    max_decode_length = input_length + self.hparams.extra_decode_length

    symbols_to_logits_fn = self._get_symbols_to_logits_fn(max_decode_length)

    # Create initial set of IDs that will be passed into symbols_to_logits_fn.
    initial_ids = tf.zeros([batch_size], dtype=tf.int32)

    # Create cache storing decoder attention values for each layer.
    cache = {
      "layer_%d" % layer: {
        "k": tf.zeros([batch_size, 0, self.hparams.hidden_dim]),
        "v": tf.zeros([batch_size, 0, self.hparams.hidden_dim]),
      } for layer in range(self.depth)}

    # Add encoder output and attention bias to the cache.
    cache["encoder_outputs"] = encoder_outputs
    tf.logging.info("cache_encode_outputs")
    tf.logging.info(cache["encoder_outputs"])
    cache["encoder_decoder_attention_bias"] = encoder_decoder_attention_bias

    # Use beam search to find the top beam_size sequences and scores.
    decoded_ids, scores = sequence_beam_search(
      symbols_to_logits_fn=symbols_to_logits_fn,
      initial_ids=initial_ids,
      initial_cache=cache,
      vocab_size=self.hparams.vocab_size,
      beam_size=self.hparams.beam_size,
      alpha=self.hparams.alpha,
      max_decode_length=max_decode_length,
      eos_id=EOS_ID)

    # Get the top sequence for each batch element
    top_decoded_ids = decoded_ids[:, 0, 1:]
    top_scores = scores[:, 0]

    return {"outputs": top_decoded_ids, "scores": top_scores}


class UniversalTransformer(Transformer):

  def __init__(self, hparams, scope="Transformer"):
    super(UniversalTransformer, self).__init__(hparams, scope)

  def create_vars(self, reuse=False):
    self.initializer = tf.variance_scaling_initializer(
      self.initializer_gain, mode="fan_avg", distribution="uniform")

    with tf.variable_scope(self.scope, initializer=self.initializer, reuse=tf.AUTO_REUSE):
      self.embedding_softmax_layer = EmbeddingSharedWeights(vocab_size=self.vocab_size, embedding_dim=self.hidden_dim,
                                                       method="matmul" if tpu else "gather")

      self.encoder_stack = UniversalTransformerEncoder(self.hidden_dim, self.number_of_heads, self.depth, self.ff_filter_size,
                                              self.dropout_keep_prob,
                                              scope="TransformerEncoder")
      self.decoder_stack = UniversalTransformerDecoder(self.hidden_dim, self.number_of_heads, self.depth, self.ff_filter_size,
                                              self.dropout_keep_prob,
                                              scope="TransformerDecoder")

      self.embedding_softmax_layer.create_vars()
      self.encoder_stack.create_vars(reuse=tf.AUTO_REUSE)
      self.decoder_stack.create_vars(reuse=tf.AUTO_REUSE)


if __name__ == '__main__':
  from distill.data_util.prep_algorithmic import AlgorithmicIdentityBinary40

  tf.logging.set_verbosity(tf.logging.INFO)

  bin_iden = AlgorithmicIdentityBinary40('data/alg')

  dataset = tf.data.TFRecordDataset(bin_iden.get_tfrecord_path(mode="train"))
  dataset = dataset.map(bin_iden.parse_examples)
  dataset = dataset.padded_batch(1, padded_shapes=bin_iden.get_padded_shapes())
  iterator = dataset.make_initializable_iterator()

  example = iterator.get_next()
  inputs, targets, lengths = example


  transformer = Transformer(hidden_dim=32,
                            number_of_heads=1,
                            depth=1,
                            ff_filter_size=10,
                            dropout_keep_prob=1.0,
                            initializer_gain=0.5,
                            scope="Transformer")
  transformer.create_vars(reuse=False)

  logits = transformer.apply(inputs=inputs,
                              targets=targets)
  predictions = tf.argmax(logits, axis=-1)

  global_step = tf.train.get_or_create_global_step()
  scaffold = tf.train.Scaffold(local_init_op=tf.group(tf.local_variables_initializer(),
                                                      iterator.initializer))
  with tf.train.MonitoredTrainingSession(checkpoint_dir='logs', scaffold=scaffold) as sess:
    print(sess.run([inputs, predictions]))
