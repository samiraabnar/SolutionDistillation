class TransformerHparam(object):
  def __init__(self, input_dim,
               hidden_dim,
               output_dim,
               depth,
               batch_size,
               pretrained_embedding_path,
               input_dropout_keep_prob,
               hidden_dropout_keep_prob,
               number_of_heads,
               ff_filter_size,
               initializer_gain,
               vocab_size,
               label_smoothing,
               ):
    self.input_dim = input_dim
    self.vocab_size = vocab_size
    self.hidden_dim = hidden_dim
    self.output_dim = output_dim
    self.depth = depth
    self.batch_size = batch_size
    self.pretrained_embedding_path = pretrained_embedding_path
    self.input_dropout_keep_prob = input_dropout_keep_prob
    self.hidden_dropout_keep_prob = hidden_dropout_keep_prob
    self.number_of_heads = number_of_heads
    self.ff_filter_size = ff_filter_size
    self.initializer_gain = initializer_gain
    self.label_smoothing = label_smoothing
    self.clip_grad_norm = 0.  # i.e. no gradient clipping
    self.optimizer_adam_epsilon = 1e-9
    self.learning_rate = 0.001
    self.learning_rate_warmup_steps = 1000
    self.initializer_gain = 1.0
    self.num_hidden_layers = depth
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

class LSTMHparam(object):
  def __init__(self, input_dim,
               hidden_dim,
               output_dim,
               depth,
               batch_size,
               pretrained_embedding_path,
               input_dropout_keep_prob,
               hidden_dropout_keep_prob,
               number_of_heads,
               ff_filter_size,
               initializer_gain,
               vocab_size,
               label_smoothing,
               attention_mechanism,
               sent_rep_mode,
               embedding_dim,
               ):
    self.input_dim = input_dim
    self.vocab_size = vocab_size
    self.hidden_dim = hidden_dim
    self.output_dim = output_dim
    self.embedding_dim = embedding_dim
    self.depth = depth
    self.batch_size = batch_size
    self.pretrained_embedding_path = pretrained_embedding_path
    self.input_dropout_keep_prob = 0.85
    self.hidden_dropout_keep_prob = 0.5
    self.number_of_heads = number_of_heads
    self.ff_filter_size = ff_filter_size
    self.initializer_gain = initializer_gain
    self.label_smoothing = label_smoothing
    self.attention_mechanism = attention_mechanism
    self.sent_rep_mode = "all"
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

