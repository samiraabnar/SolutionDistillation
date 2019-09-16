class TransformerHparam(object):
  def __init__(self, input_dim,
               hidden_dim,
               output_dim,
               encoder_depth,
               decoder_depth,
               batch_size,
               input_dropout_keep_prob,
               hidden_dropout_keep_prob,
               number_of_heads,
               ff_filter_size,
               initializer_gain,
               vocab_size,
               label_smoothing,
               train_embeddings,
               learning_rate,
               encoder_self_attention_dir="top_down",
               decoder_self_attention_dir="top_down",
               decoder_cross_attention_dir="top_down",
               attention_mechanism=None,
               sent_rep_mode=None,
               embedding_dim=None,
               cls_token=False,
               attention_dropout_keepprob=1.0,
               relu_dropout_keepprob=1.0,
               postprocess_dropout_keepprob=1.0
               ):
    self.input_dim = input_dim
    self.vocab_size = vocab_size
    self.hidden_dim = hidden_dim
    self.output_dim = output_dim
    self.encoder_depth = encoder_depth
    self.decoder_depth = decoder_depth
    self.batch_size = batch_size
    self.input_dropout_keep_prob = input_dropout_keep_prob
    self.hidden_dropout_keep_prob = hidden_dropout_keep_prob
    self.number_of_heads = number_of_heads
    self.ff_filter_size = ff_filter_size
    self.initializer_gain = initializer_gain
    self.label_smoothing = label_smoothing
    self.optimizer_adam_epsilon = 1e-9
    self.learning_rate = learning_rate
    self.learning_rate_warmup_steps = 1000
    self.initializer_gain = 1.0
    self.initializer = "uniform_unit_scaling"
    self.weight_decay = 0.0
    self.optimizer_adam_beta1 = 0.9
    self.optimizer_adam_beta2 = 0.98
    self.num_sampled_classes = 0
    self.label_smoothing = 0.1
    self.clip_grad_norm = 5  # 0 i.e. no gradient clipping
    self.optimizer_adam_epsilon = 1e-9
    self.alpha = 1
    self.beam_size = 5
    self.extra_decode_length = 5
    self.encoder_self_attention_dir = encoder_self_attention_dir
    self.decoder_self_attention_dir = decoder_self_attention_dir
    self.decoder_cross_attention_dir = decoder_cross_attention_dir
    self.train_embeddings = train_embeddings
    self.cls_token = cls_token
    self.attention_dropout_keepprob = attention_dropout_keepprob
    self.relu_dropout_keepprob = relu_dropout_keepprob
    self.postprocess_dropout_keepprob = postprocess_dropout_keepprob


class LSTMHparam(object):
  def __init__(self, input_dim,
               hidden_dim,
               output_dim,
               encoder_depth,
               decoder_depth,
               batch_size,
               input_dropout_keep_prob,
               hidden_dropout_keep_prob,
               number_of_heads,
               ff_filter_size,
               initializer_gain,
               vocab_size,
               label_smoothing,
               embedding_dim,
               train_embeddings,
               learning_rate,
               encoder_self_attention_dir="top_down",
               decoder_self_attention_dir="top_down",
               decoder_cross_attention_dir="top_down",
               attention_mechanism=None,
               sent_rep_mode=None,
               cls_token=False,
               attention_dropout_keepprob=1.0,
               relu_dropout_keepprob=1.0,
               postprocess_dropout_keepprob=1.0
               ):
    self.input_dim = input_dim
    self.vocab_size = vocab_size
    self.hidden_dim = hidden_dim
    self.output_dim = output_dim
    self.embedding_dim = embedding_dim
    self.encoder_depth = encoder_depth
    self.decoder_depth = decoder_depth
    self.batch_size = batch_size
    self.input_dropout_keep_prob = input_dropout_keep_prob
    self.hidden_dropout_keep_prob = hidden_dropout_keep_prob
    self.number_of_heads = number_of_heads
    self.ff_filter_size = ff_filter_size
    self.initializer_gain = initializer_gain
    self.attention_mechanism = attention_mechanism
    self.sent_rep_mode = "final"
    self.clip_grad_norm = 5  # i.e. 9 is no gradient clipping
    self.optimizer_adam_epsilon = 1e-9
    self.learning_rate = learning_rate
    self.learning_rate_warmup_steps = 2000
    self.initializer_gain = 1.0
    self.initializer = "uniform_unit_scaling"
    self.weight_decay = 0.0
    self.optimizer_adam_beta1 = 0.9
    self.optimizer_adam_beta2 = 0.98
    self.num_sampled_classes = 0
    self.label_smoothing = label_smoothing
    self.optimizer_adam_epsilon = 1e-9
    self.train_embeddings = train_embeddings
    self.attention_dropout_keepprob = attention_dropout_keepprob
    self.relu_dropout_keepprob = relu_dropout_keepprob
    self.postprocess_dropout_keepprob = postprocess_dropout_keepprob


class LenetHparams(object):
  def __init__(self, input_dim,
               hidden_dim,
               output_dim,
               encoder_depth,
               decoder_depth,
               batch_size,
               input_dropout_keep_prob,
               hidden_dropout_keep_prob,
               number_of_heads,
               ff_filter_size,
               initializer_gain,
               vocab_size,
               label_smoothing,
               embedding_dim,
               train_embeddings,
               learning_rate,
               encoder_self_attention_dir="top_down",
               decoder_self_attention_dir="top_down",
               decoder_cross_attention_dir="top_down",
               attention_mechanism=None,
               sent_rep_mode=None,
               cls_token=False,
               attention_dropout_keepprob=1.0,
               relu_dropout_keepprob=1.0,
               postprocess_dropout_keepprob=1.0
               ):
    self.input_dim = input_dim
    self.vocab_size = vocab_size
    self.hidden_dim = hidden_dim
    self.output_dim = output_dim
    self.batch_size = batch_size
    self.input_dropout_keep_prob = input_dropout_keep_prob
    self.hidden_dropout_keep_prob = hidden_dropout_keep_prob
    self.initializer_gain = initializer_gain
    self.sent_rep_mode = "final"
    self.clip_grad_norm = 5  # i.e. 9 is no gradient clipping
    self.optimizer_adam_epsilon = 1e-9
    self.learning_rate = learning_rate
    self.learning_rate_warmup_steps = 2000
    self.initializer_gain = 1.0
    self.initializer = "uniform_unit_scaling"
    self.weight_decay = 0.0
    self.optimizer_adam_beta1 = 0.9
    self.optimizer_adam_beta2 = 0.98
    self.num_sampled_classes = 0
    self.label_smoothing = label_smoothing
    self.optimizer_adam_epsilon = 1e-9
    self.filter_size=5
    self.encoder_depth = encoder_depth
