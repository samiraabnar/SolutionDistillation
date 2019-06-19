import tensorflow as tf
import numpy as np
from tensor2tensor.utils.beam_search import EOS_ID

from distill.common.beam_search import sequence_beam_search
from distill.common.layer_utils import get_decoder_self_attention_bias, get_position_encoding, get_padding_bias, get_padding
from distill.data_util.prep_arithmatic import Arithmatic
from distill.data_util.prep_sst import SST
from distill.layers.attention import MultiHeadScaledDotProductAttention, ReversedMultiHeadScaledDotProductAttention
from distill.layers.embedding import EmbeddingSharedWeights
from distill.layers.ffn_layer import FeedFowardNetwork
from distill.layers.pre_post_wrapper import PrePostProcessingWrapper, LayerNormalization

tpu = False


class TransformerEncoder(object):
  def __init__(self, hidden_dim, number_of_heads, depth, ff_filter_size, dropout_keep_prob, self_attention_dir="top_down", scope="TransformerEncoder"):
    self.hidden_dim = hidden_dim
    self.number_of_heads = number_of_heads
    self.depth = depth
    self.dropout_keep_prob = dropout_keep_prob
    self.ff_filter_size = ff_filter_size
    self.self_attention_dir = self_attention_dir

    self.scope = scope

  def create_vars(self, reuse=False):

    with tf.variable_scope(self.scope, reuse=reuse):
      self.layers = []
      for i in np.arange(self.depth):
        # Create sublayers for each layer.

        if self.self_attention_dir == "bottom_up":
          self_attention_layer = ReversedMultiHeadScaledDotProductAttention(hidden_dim=self.hidden_dim,
                                                                            num_heads=self.number_of_heads,
                                                                            attention_dropout_keepprob=self.dropout_keep_prob,
                                                                            scope="Attention" + str(i))
        else:
          self_attention_layer = MultiHeadScaledDotProductAttention(hidden_dim=self.hidden_dim,
                                                                            num_heads=self.number_of_heads,
                                                                            attention_dropout_keepprob=self.dropout_keep_prob,
                                                                            scope="Attention" + str(i))

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

  def apply(self, inputs, attention_bias, inputs_padding, is_train=True, encoder_inputs_presence=None, reuse=tf.AUTO_REUSE):

    encoder_inputs = inputs
    with tf.variable_scope(self.scope, reuse=reuse):
      for n, layer in enumerate(self.layers):
        # Run inputs through the sublayers.
        self_attention_layer = layer[0]
        feed_forward_network = layer[1]

        tf.logging.info("encoder inputs:")
        tf.logging.info(encoder_inputs)

        encoder_inputs, encoder_inputs_presence = self_attention_layer.apply(x=encoder_inputs, y=encoder_inputs,
                                                                             x_presence=encoder_inputs_presence,
                                                                             y_presence=encoder_inputs_presence,
                                                                             is_train=is_train, bias=attention_bias)
        encoder_inputs, _ = feed_forward_network.apply(x=encoder_inputs, is_train=is_train,
                                                    padding=inputs_padding)

    return self.output_normalization.apply(encoder_inputs, is_train), encoder_inputs_presence


class TransformerDecoder(object):
  def __init__(self, hidden_dim, number_of_heads, depth, ff_filter_size, dropout_keep_prob, self_attention_dir="top_down", cross_attention_dir="top_down", scope="TransformerDecoder"):
    self.hidden_dim = hidden_dim
    self.number_of_heads = number_of_heads
    self.depth = depth
    self.dropout_keep_prob = dropout_keep_prob
    self.ff_filter_size = ff_filter_size
    self.self_attention_dir = self_attention_dir
    self.cross_attention_dir = cross_attention_dir
    self.scope = scope

  def create_vars(self, reuse=False):

    with tf.variable_scope(self.scope, reuse=reuse):
      self.layers = []
      for i in np.arange(self.depth):
        # Create sublayers for each layer.
        if self.self_attention_dir == "bottom_up":
          self_attention_layer = ReversedMultiHeadScaledDotProductAttention(hidden_dim=self.hidden_dim,
                                                                  num_heads=self.number_of_heads,
                                                                  attention_dropout_keepprob=self.dropout_keep_prob,
                                                                  scope="SelfAttention"+str(i))
        else:
          self_attention_layer = MultiHeadScaledDotProductAttention(hidden_dim=self.hidden_dim,
                                                                    num_heads=self.number_of_heads,
                                                                    attention_dropout_keepprob=self.dropout_keep_prob,
                                                                    scope="SelfAttention" + str(i))

        if self.cross_attention_dir == "bottom_up":
          enc_dec_attention_layer = ReversedMultiHeadScaledDotProductAttention(hidden_dim=self.hidden_dim,
                                                                  num_heads=self.number_of_heads,
                                                                  attention_dropout_keepprob=self.dropout_keep_prob,
                                                                  scope="EncDecAttention" + str(i))
        else:
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

  def apply(self, inputs, encoder_outputs, decoder_self_attention_bias, attention_bias, encoder_outputs_presence=None,
            cache=None, is_train=True,  reuse=tf.AUTO_REUSE,
            target_length=None):
    decoder_inputs = inputs
    with tf.variable_scope(self.scope, reuse=reuse):
      for n, layer in enumerate(self.layers):
        layer_name = "layer_%d" % n
        layer_cache = cache[layer_name] if cache is not None else None

        # Run inputs through the sublayers.
        self_attention_layer = layer[0]
        enc_dec_attention = layer[1]
        feed_forward_network = layer[2]

        decoder_inputs, _ = self_attention_layer.apply(x=decoder_inputs, y=decoder_inputs, is_train=is_train,
                                                      bias=decoder_self_attention_bias, cache=layer_cache)
        decoder_inputs, _ = enc_dec_attention.apply(x=decoder_inputs, y=encoder_outputs, is_train=is_train,
                                                       y_presence=encoder_outputs_presence,
                                                       bias=attention_bias)
        decoder_inputs,_ = feed_forward_network.apply(x=decoder_inputs, is_train=is_train)

    return self.output_normalization.apply(decoder_inputs, is_train)


class UniversalTransformerEncoder(TransformerEncoder):
  def __init__(self, hidden_dim, number_of_heads, depth, ff_filter_size, dropout_keep_prob, self_attention_dir="top_down", scope="TransformerEncoder"):
    super(UniversalTransformerEncoder, self).__init__(hidden_dim, number_of_heads, depth, ff_filter_size, dropout_keep_prob, self_attention_dir, scope)

  def create_vars(self, reuse=False):

    with tf.variable_scope(self.scope, reuse=reuse):
      self.layers = []
      for i in np.arange(self.depth):
        # Create sublayers for each layer.
        if self.self_attention_dir == "bottom_up":
          self_attention_layer = ReversedMultiHeadScaledDotProductAttention(hidden_dim=self.hidden_dim,
                                                                            num_heads=self.number_of_heads,
                                                                            attention_dropout_keepprob=self.dropout_keep_prob,
                                                                            scope="Attention")
        else:
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
  def __init__(self, hidden_dim, number_of_heads, depth, ff_filter_size, dropout_keep_prob,
               self_attention_dir="top_down", cross_attention_dir="top_down", scope="TransformerDecoder"):
    super(UniversalTransformerDecoder, self).__init__(hidden_dim, number_of_heads, depth, ff_filter_size,
                                                      dropout_keep_prob,self_attention_dir,cross_attention_dir, scope)


  def create_vars(self, reuse=False):

    with tf.variable_scope(self.scope, reuse=reuse):
      self.layers = []
      for i in np.arange(self.depth):
        # Create sublayers for each layer.
        if self.self_attention_dir == "bottom_up":
          self_attention_layer = ReversedMultiHeadScaledDotProductAttention(hidden_dim=self.hidden_dim,
                                                                  num_heads=self.number_of_heads,
                                                                  attention_dropout_keepprob=self.dropout_keep_prob,
                                                                  scope="SelfAttention")
        else:
          self_attention_layer = MultiHeadScaledDotProductAttention(hidden_dim=self.hidden_dim,
                                                                    num_heads=self.number_of_heads,
                                                                    attention_dropout_keepprob=self.dropout_keep_prob,
                                                                    scope="SelfAttention")

        if self.cross_attention_dir == "bottom_up":
          enc_dec_attention_layer = ReversedMultiHeadScaledDotProductAttention(hidden_dim=self.hidden_dim,
                                                                  num_heads=self.number_of_heads,
                                                                  attention_dropout_keepprob=self.dropout_keep_prob,
                                                                  scope="EncDecAttention")
        else:
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

  def __init__(self, hparams, task, scope="Transformer"):
    self.hparams = hparams
    self.vocab_size = hparams.vocab_size
    self.hidden_dim = hparams.hidden_dim
    self.number_of_heads = hparams.number_of_heads
    self.decoder_depth = hparams.decoder_depth
    self.encoder_depth = hparams.encoder_depth
    self.ff_filter_size = hparams.ff_filter_size
    self.dropout_keep_prob = hparams.hidden_dropout_keep_prob
    self.initializer_gain = hparams.initializer_gain
    self.scope = scope
    self.task = task
    self.eos_id = self.task.eos_id

  def create_vars(self, reuse=False,pretrained_embeddings=None):
    self.initializer = tf.variance_scaling_initializer(
      self.initializer_gain, mode="fan_avg", distribution="uniform")

    with tf.variable_scope(self.scope, initializer=self.initializer, reuse=tf.AUTO_REUSE):

      self.input_embedding_layer = EmbeddingSharedWeights(vocab_size=self.vocab_size, embedding_dim=self.hidden_dim,
                                                       method="matmul" if tpu else "gather", scope="InputEmbed",
                                                          pretrained_embeddings=pretrained_embeddings)
      self.input_embedding_layer.create_vars(is_train=self.hparams.train_embeddings)
      if not self.task.share_input_output_embeddings:
        self.output_embedding_layer = EmbeddingSharedWeights(vocab_size=len(self.task.target_vocab),
                                       embedding_dim=self.hparams.hidden_dim, scope="OutputEmbed")
        self.output_embedding_layer.create_vars()
      else:
        self.output_embedding_layer = self.input_embedding_layer

      self.encoder_stack = TransformerEncoder(self.hidden_dim, self.number_of_heads, self.encoder_depth, self.ff_filter_size,
                                              self.dropout_keep_prob,
                                              self_attention_dir=self.hparams.encoder_self_attention_dir,
                                              scope="TransformerEncoder")
      self.decoder_stack = TransformerDecoder(self.hidden_dim, self.number_of_heads, self.decoder_depth, self.ff_filter_size,
                                              self.dropout_keep_prob,
                                              self_attention_dir=self.hparams.decoder_self_attention_dir,
                                              cross_attention_dir=self.hparams.decoder_cross_attention_dir,
                                              scope="TransformerDecoder")


      self.encoder_stack.create_vars(reuse=False)
      self.decoder_stack.create_vars(reuse=False)

  def apply(self, examples, target_length=None, reuse=tf.AUTO_REUSE, is_train=True):
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

    with tf.variable_scope(self.scope, initializer=self.initializer, reuse=reuse):
      # Calculate attention bias for encoder self-attention and decoder
      # multi-headed attention layers.
      attention_bias = get_padding_bias(inputs)

      # Run the inputs through the encoder layer to map the symbol
      # representations to continuous representations.
      encoder_outputs, encoder_outputs_presence = self.encode(inputs, attention_bias, is_train)
      tf.logging.info('encoder outputs')
      tf.logging.info(encoder_outputs)

      if targets is None or not is_train:
        output_dic = self.predict(encoder_outputs=encoder_outputs,
                                  encoder_outputs_presence=encoder_outputs_presence,
                                  encoder_decoder_attention_bias=attention_bias,
                                  target_length=target_length)
        predictions = output_dic['outputs']
        logits = tf.one_hot(indices=predictions, depth=len(self.task.target_vocab))
        tf.logging.info('predict logits')
        tf.logging.info(logits)
        outputs = self.output_embedding_layer.apply(tf.cast(logits, dtype=tf.int32))
      else:
        outputs = self.decode(targets, encoder_outputs=encoder_outputs,
                              encoder_outputs_presence=encoder_outputs_presence,
                              attention_bias=attention_bias,
                              is_train=is_train,
                              target_length=target_length)
        logits = self.output_embedding_layer.linear(outputs)

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
      embedded_inputs = self.input_embedding_layer.apply(inputs)
      inputs_padding = get_padding(inputs)

      with tf.name_scope("add_pos_encoding"):
        length = tf.shape(embedded_inputs)[1]
        pos_encoding = get_position_encoding(
            length, self.hidden_dim)
        encoder_inputs = embedded_inputs + pos_encoding

      if is_train:
        encoder_inputs = tf.nn.dropout(
            encoder_inputs, keep_prob=self.hparams.input_dropout_keep_prob)

      return self.encoder_stack.apply(encoder_inputs, attention_bias, inputs_padding, is_train)

  def decode(self, targets, encoder_outputs, attention_bias, encoder_outputs_presence=None, is_train=True,target_length=None):
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
      decoder_inputs = self.output_embedding_layer.apply(targets)
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
            decoder_inputs, keep_prob=self.dropout_keep_prob)

      # Run values
      decoder_self_attention_bias = get_decoder_self_attention_bias(
          length)


      outputs = self.decoder_stack.apply(
            inputs=decoder_inputs,
            encoder_outputs=encoder_outputs,
            decoder_self_attention_bias=decoder_self_attention_bias,
            attention_bias=attention_bias,
            encoder_outputs_presence=encoder_outputs_presence,
            target_length=None)


      return outputs

  def _get_symbols_to_logits_fn(self, max_decode_length, encoder_outputs_presence=None):
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
      decoder_input = self.output_embedding_layer.apply(decoder_input)
      decoder_input += timing_signal[i:i + 1]

      self_attention_bias = decoder_self_attention_bias[:, :, i:i + 1, :i + 1]

      decoder_outputs = self.decoder_stack.apply(
        inputs=decoder_input, encoder_outputs=cache.get("encoder_outputs"),
        decoder_self_attention_bias=self_attention_bias,
        attention_bias=cache.get("encoder_decoder_attention_bias"),
        encoder_outputs_presence=encoder_outputs_presence,
        cache=cache, is_train=False)
      logits = self.output_embedding_layer.linear(decoder_outputs)

      logits = tf.squeeze(logits, axis=[1])
      return logits, cache

    return symbols_to_logits_fn

  def predict(self, encoder_outputs, encoder_decoder_attention_bias, encoder_outputs_presence=None, target_length=None):
    """Return predicted sequence."""
    batch_size = tf.shape(encoder_outputs)[0]
    input_length = tf.shape(encoder_outputs)[1]
    if target_length is None:
      max_decode_length = input_length + self.hparams.extra_decode_length
    else:
      max_decode_length = target_length

    symbols_to_logits_fn = self._get_symbols_to_logits_fn(max_decode_length, encoder_outputs_presence)

    # Create initial set of IDs that will be passed into symbols_to_logits_fn.
    initial_ids = tf.zeros([batch_size], dtype=tf.int32)

    # Create cache storing decoder attention values for each layer.
    cache = {
      "layer_%d" % layer: {
        "k": tf.zeros([batch_size, 0, self.hparams.hidden_dim]),
        "v": tf.zeros([batch_size, 0, self.hparams.hidden_dim]),
      } for layer in range(self.decoder_depth)}

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
      vocab_size=len(self.task.target_vocab),
      beam_size=self.hparams.beam_size,
      alpha=self.hparams.alpha,
      max_decode_length=max_decode_length,
      eos_id=self.eos_id)

    # Get the top sequence for each batch element
    top_decoded_ids = decoded_ids[:, 0, 1:]
    top_scores = scores[:, 0]

    return {"outputs": top_decoded_ids, "scores": top_scores}


class UniversalTransformer(Transformer):

  def __init__(self, hparams, task, scope="Transformer"):
    super(UniversalTransformer, self).__init__(hparams, task, scope)

  def create_vars(self, reuse=False, pretrained_embeddings=None):
    self.initializer = tf.variance_scaling_initializer(
      self.initializer_gain, mode="fan_avg", distribution="uniform")

    with tf.variable_scope(self.scope, initializer=self.initializer, reuse=tf.AUTO_REUSE):
      self.input_embedding_layer = EmbeddingSharedWeights(vocab_size=self.vocab_size, embedding_dim=self.hidden_dim,
                                                          method="matmul" if tpu else "gather", scope="InputEmbed",
                                                          pretrained_embeddings=pretrained_embeddings)
      self.input_embedding_layer.create_vars(is_train=self.hparams.train_embeddings)
      if not self.task.share_input_output_embeddings:
        self.output_embedding_layer = EmbeddingSharedWeights(vocab_size=len(self.task.target_vocab),
                                                             embedding_dim=self.hparams.hidden_dim, scope="OutputEmbed")
        self.output_embedding_layer.create_vars()
      else:
        self.output_embedding_layer = self.input_embedding_layer

      self.encoder_stack = UniversalTransformerEncoder(self.hidden_dim, self.number_of_heads, self.encoder_depth, self.ff_filter_size,
                                              self.dropout_keep_prob,
                                              scope="TransformerEncoder")
      self.decoder_stack = UniversalTransformerDecoder(self.hidden_dim, self.number_of_heads, self.decoder_depth, self.ff_filter_size,
                                              self.dropout_keep_prob,
                                              scope="TransformerDecoder")

      self.encoder_stack.create_vars(reuse=tf.AUTO_REUSE)
      self.decoder_stack.create_vars(reuse=tf.AUTO_REUSE)

class EncodingTransformer(object):
  """Transformer model for sequence to sequence data.
  Implemented as described in: https://arxiv.org/pdf/1706.03762.pdf
  The Transformer model consists of an encoder and decoder. The input is an int
  sequence (or a batch of sequences). The encoder produces a continous
  representation, and the decoder uses the encoder output to generate
  probabilities for the output sequence.
  """

  def __init__(self, hparams, task, scope="Transformer"):
    self.hparams = hparams
    self.vocab_size = hparams.vocab_size
    self.hidden_dim = hparams.hidden_dim
    self.number_of_heads = hparams.number_of_heads
    self.encoder_depth = hparams.encoder_depth
    self.ff_filter_size = hparams.ff_filter_size
    self.dropout_keep_prob = hparams.hidden_dropout_keep_prob
    self.initializer_gain = hparams.initializer_gain
    self.scope = scope
    self.task = task
    self.eos_id = self.task.eos_id

  def create_vars(self, reuse=False,pretrained_embeddings=None):
    self.initializer = tf.variance_scaling_initializer(
      self.initializer_gain, mode="fan_avg", distribution="normal")

    with tf.variable_scope(self.scope, initializer=self.initializer, reuse=tf.AUTO_REUSE):

      self.input_embedding_layer = EmbeddingSharedWeights(vocab_size=self.vocab_size, embedding_dim=self.hidden_dim,
                                                       method="matmul" if tpu else "gather", scope="InputEmbed",
                                                          pretrained_embeddings=pretrained_embeddings)
      self.input_embedding_layer.create_vars(is_train=self.hparams.train_embeddings)
      if not self.task.share_input_output_embeddings:
        self.output_embedding_layer = EmbeddingSharedWeights(vocab_size=len(self.task.target_vocab),
                                       embedding_dim=self.hparams.hidden_dim, scope="OutputEmbed")
        self.output_embedding_layer.create_vars()
      else:
        self.output_embedding_layer = self.input_embedding_layer
    
#      self.output_projections_layer = tf.layers.Dense(len(self.task.target_vocab),
#                                                      activation=None,
#                                                      use_bias=True,
#                                                      kernel_initializer=self.initializer,
#                                                      bias_initializer=tf.zeros_initializer(),
#                                                      kernel_regularizer=None,
#                                                      bias_regularizer=None,
#                                                      activity_regularizer=None,
#                                                      kernel_constraint=None,
#                                                      bias_constraint=None,
#                                                      trainable=True,
#                                                      name="OutProj")

      self.encoder_stack = TransformerEncoder(self.hidden_dim, self.number_of_heads, self.encoder_depth, self.ff_filter_size,
                                              self.dropout_keep_prob,
                                              self_attention_dir=self.hparams.encoder_self_attention_dir,
                                              scope="TransformerEncoder")


      self.encoder_stack.create_vars(reuse=False)

  def apply(self, examples, target_length=None, reuse=tf.AUTO_REUSE, is_train=True):
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

    with tf.variable_scope(self.scope, initializer=self.initializer, reuse=reuse):
      # Calculate attention bias for encoder self-attention and decoder
      # multi-headed attention layers.
      attention_bias = get_padding_bias(inputs)

      # Run the inputs through the encoder layer to map the symbol
      # representations to continuous representations.
      encoder_outputs, encoder_outputs_presence = self.encode(inputs, attention_bias, is_train)
      tf.logging.info('encoder outputs')
      tf.logging.info(encoder_outputs)

      outputs = self.decode(encoder_outputs=encoder_outputs,
                            encoder_outputs_presence=encoder_outputs_presence,
                            is_train=is_train)

      logits =  self.output_embedding_layer.linear(outputs)
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
      embedded_inputs = self.input_embedding_layer.apply(inputs)
      inputs_padding = get_padding(inputs)

      with tf.name_scope("add_pos_encoding"):
        length = tf.shape(embedded_inputs)[1]
        pos_encoding = get_position_encoding(
            length, self.hidden_dim)
        encoder_inputs = embedded_inputs + pos_encoding

      if is_train:
        encoder_inputs = tf.nn.dropout(
            encoder_inputs, keep_prob=self.hparams.input_dropout_keep_prob)

      return self.encoder_stack.apply(encoder_inputs, attention_bias, inputs_padding, is_train)

  def decode(self,encoder_outputs, encoder_outputs_presence=None, is_train=True):
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
      if encoder_outputs_presence is not None:
        outputs = tf.reduce_sum(encoder_outputs * encoder_outputs_presence, axis=1)
      else:
        outputs = tf.reduce_mean(encoder_outputs, axis=1)


      return tf.expand_dims(outputs, axis=1)


class EncodingUniversalTransformer(EncodingTransformer):
  """Transformer model for sequence to sequence data.
  Implemented as described in: https://arxiv.org/pdf/1706.03762.pdf
  The Transformer model consists of an encoder and decoder. The input is an int
  sequence (or a batch of sequences). The encoder produces a continous
  representation, and the decoder uses the encoder output to generate
  probabilities for the output sequence.
  """

  def __init__(self, hparams, task, scope="EncUTransformer"):
    self.hparams = hparams
    self.vocab_size = hparams.vocab_size
    self.hidden_dim = hparams.hidden_dim
    self.number_of_heads = hparams.number_of_heads
    self.encoder_depth = hparams.encoder_depth
    self.ff_filter_size = hparams.ff_filter_size
    self.dropout_keep_prob = hparams.hidden_dropout_keep_prob
    self.initializer_gain = hparams.initializer_gain
    self.scope = scope
    self.task = task
    self.eos_id = self.task.eos_id

  def create_vars(self, reuse=False,pretrained_embeddings=None):
    self.initializer = tf.variance_scaling_initializer(
      self.initializer_gain, mode="fan_avg", distribution="uniform")

    with tf.variable_scope(self.scope, initializer=self.initializer, reuse=tf.AUTO_REUSE):

      self.input_embedding_layer = EmbeddingSharedWeights(vocab_size=self.vocab_size, embedding_dim=self.hidden_dim,
                                                       method="matmul" if tpu else "gather", scope="InputEmbed",
                                                          pretrained_embeddings=pretrained_embeddings)
      self.input_embedding_layer.create_vars(is_train=self.hparams.train_embeddings)
      if not self.task.share_input_output_embeddings:
        self.output_embedding_layer = EmbeddingSharedWeights(vocab_size=len(self.task.target_vocab),
                                       embedding_dim=self.hparams.hidden_dim, scope="OutputEmbed")
        self.output_embedding_layer.create_vars()
      else:
        self.output_embedding_layer = self.input_embedding_layer

      self.encoder_stack = UniversalTransformerEncoder(self.hidden_dim, self.number_of_heads, self.encoder_depth, self.ff_filter_size,
                                              self.dropout_keep_prob,
                                              self_attention_dir=self.hparams.encoder_self_attention_dir,
                                              scope="UniversalTransformerEncoder")


      self.encoder_stack.create_vars(reuse=tf.AUTO_REUSE)


if __name__ == '__main__':
  from distill.data_util.prep_algorithmic import AlgorithmicIdentityBinary40

  tf.logging.set_verbosity(tf.logging.INFO)

  #bin_iden = AlgorithmicIdentityBinary40('data/alg')
  #bin_iden = Arithmatic('data/arithmatic')
  bin_iden = SST(data_path="data/sst/",
      add_subtrees=False,
      pretrained=True)

  dataset = tf.data.TFRecordDataset(bin_iden.get_tfrecord_path(mode="train"))
  dataset = dataset.map(bin_iden.parse_examples)
  dataset = dataset.padded_batch(32, padded_shapes=bin_iden.get_padded_shapes())
  iterator = dataset.make_initializable_iterator()

  example = iterator.get_next()
  inputs, targets, inputs_lengths, targets_length = example

  class Config(object):
    def __init__(self):
      self.vocab_size = bin_iden.vocab_length
      self.hidden_dim = 32
      self.output_dim = self.vocab_size
      self.embedding_dim = 32
      self.input_dropout_keep_prob = 0.5
      self.hidden_dropout_keep_prob = 0.5
      self.attention_mechanism = None
      self.encoder_depth = 1
      self.decoder_depth = 1
      self.sent_rep_mode = "all"
      self.scope = "transformer"
      self.batch_size = 64
      self.input_dropout_keep_prob = 1.0
      self.hidden_dropout_keep_prob = 1.0
      self.number_of_heads = 2
      self.ff_filter_size = 512
      self.initializer_gain = 1.0
      self.label_smoothing = 0.1
      self.clip_grad_norm = 0.  # i.e. no gradient clipping
      self.optimizer_adam_epsilon = 1e-9
      self.learning_rate = 0.001
      self.learning_rate_warmup_steps = 1000
      self.initializer_gain = 1.0
      self.initializer = "uniform_unit_scaling"
      self.weight_decay = 0.0
      self.optimizer_adam_beta1 = 0.9
      self.optimizer_adam_beta2 = 0.98
      self.num_sampled_classes = 0
      self.label_smoothing = 0.1
      self.clip_grad_norm = 0.  # i.e. no gradient clipping
      self.optimizer_adam_epsilon = 1e-9
      self.alpha = 1
      self.beam_size = 5
      self.extra_decode_length = 5
      self.encoder_self_attention_dir = "bottom_up"
      self.decoder_self_attention_dir = "top_down"
      self.decoder_cross_attention_dir = "top_down"
      self.train_embeddings = True


  transformer = EncodingTransformer(Config(),
                            task=bin_iden,
                            scope="Transformer")
  transformer.create_vars(reuse=False)

  outputs = transformer.apply(example, target_length=bin_iden.target_length, is_train=True)
  outputs = transformer.apply(example, target_length=bin_iden.target_length, is_train=False)

  logits = outputs['logits']
  predictions = tf.argmax(logits, axis=-1)

  global_step = tf.train.get_or_create_global_step()
  scaffold = tf.train.Scaffold(local_init_op=tf.group(tf.local_variables_initializer(),
                                                      iterator.initializer))
  with tf.train.MonitoredTrainingSession(checkpoint_dir='logs/test_transformer', scaffold=scaffold) as sess:
    for _ in np.arange(10):
      inp, targ, pred = sess.run([inputs, targets, predictions])

      print(inp)
      print(bin_iden.decode(inp[0]))
      print(bin_iden.decode(pred[0]))
      print(bin_iden.decode(targ[0]))