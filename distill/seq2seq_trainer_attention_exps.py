
import tensorflow as tf
import os
from distill.common.hparams import TransformerHparam, LSTMHparam
from distill.data_util.prep_algorithmic import AlgorithmicIdentityDecimal40, AlgorithmicAdditionDecimal40, \
  AlgorithmicMultiplicationDecimal40, AlgorithmicSortProblem, AlgorithmicReverseProblem, AlgorithmicIdentityBinary40
from distill.data_util.prep_arithmatic import Arithmatic
from distill.data_util.prep_sst import SST
from distill.models.lstm_seq2seq import LSTMSeq2Seq, BidiLSTMSeq2Seq
from distill.models.transformer import Transformer, UniversalTransformer
from distill.pipelines.seq2seq import Seq2SeqTrainer

tf.logging.set_verbosity(tf.logging.INFO)

tf.app.flags.DEFINE_string("exp_name", "trial", "")
tf.app.flags.DEFINE_string("task_name", "arithmatic", "sst | arithmatic | identity_binary| identity | addition| multiplication | sort | reverse")
tf.app.flags.DEFINE_string("log_dir", "logs", "")
tf.app.flags.DEFINE_string("save_dir", None, "")
tf.app.flags.DEFINE_string("model", "transformer", "transformer | utransformer | lstm | bilstm")


tf.app.flags.DEFINE_integer("hidden_dim", 128, "")
tf.app.flags.DEFINE_integer("depth", 2, "")
tf.app.flags.DEFINE_integer("input_dim", None, "")
tf.app.flags.DEFINE_integer("output_dim", 1, "")
tf.app.flags.DEFINE_integer("number_of_heads", 4, "")
tf.app.flags.DEFINE_integer("ff_filter_size", 512, "")
tf.app.flags.DEFINE_float("initializer_gain", 1.0, "")
tf.app.flags.DEFINE_float("label_smoothing", 0.1, "")


tf.app.flags.DEFINE_float("input_dropout_keep_prob", 0.9, "")
tf.app.flags.DEFINE_float("hidden_dropout_keep_prob", 0.8, "")

tf.app.flags.DEFINE_float("learning_rate", 0.01, "")
tf.app.flags.DEFINE_float("l2_rate", 0.00005, "")


tf.app.flags.DEFINE_integer("batch_size", 128, "")
tf.app.flags.DEFINE_integer("training_iterations", 30000, "")

tf.app.flags.DEFINE_integer("vocab_size", 3, "")
tf.app.flags.DEFINE_integer("embedding_dim", 300, "embeddings dim")


tf.app.flags.DEFINE_string("pretrained_embedding_path", "data/sst/filtered_glove.txt", "pretrained embedding path")
tf.app.flags.DEFINE_string("data_path", "./data", "data path")


hparams = tf.app.flags.FLAGS


class transformer_small_hparams(TransformerHparam):
  def __init__(self, vocab_size, output_dim, input_dim, encoder_attention_dir):
    super(transformer_small_hparams, self).__init__(input_dim=input_dim,
                                                    hidden_dim=128,
                                                    output_dim=output_dim,
                                                    depth=2,
                                                    number_of_heads=2,
                                                    ff_filter_size=512,
                                                    initializer_gain=hparams.initializer_gain,
                                                    batch_size=hparams.batch_size,
                                                    input_dropout_keep_prob=0.8,
                                                    hidden_dropout_keep_prob=0.75,
                                                    vocab_size=vocab_size,
                                                    label_smoothing=hparams.label_smoothing,
                                                    encoder_self_attention_dir = encoder_attention_dir,
                                                    decoder_self_attention_dir = "top_down",
                                                    decoder_cross_attention_dir = "top_down"
                                                    )

class lstm_small_hparams(LSTMHparam):
  def __init__(self, vocab_size, output_dim, input_dim):
    super(lstm_small_hparams, self).__init__(input_dim=input_dim,
                                             hidden_dim=64,
                                             output_dim=output_dim,
                                             depth=2,
                                             number_of_heads=1,
                                             ff_filter_size=hparams.ff_filter_size,
                                             initializer_gain=hparams.initializer_gain,
                                             batch_size=hparams.batch_size,
                                             pretrained_embedding_path=hparams.pretrained_embedding_path,
                                             input_dropout_keep_prob=0.8,
                                             hidden_dropout_keep_prob=0.5,
                                             vocab_size=vocab_size,
                                             label_smoothing=hparams.label_smoothing,
                                             attention_mechanism=None,
                                             sent_rep_mode="all",
                                             embedding_dim=hparams.vocab_size / 2 if hparams.vocab_size < 100 else 100)

if __name__ == '__main__':


  Models = {"lstm": LSTMSeq2Seq,
            "bilstm": BidiLSTMSeq2Seq,
            "transformer": Transformer,
            "utransformer": UniversalTransformer}


  tasks = {'identity': AlgorithmicIdentityDecimal40('data/alg'),
           'identity_binary': AlgorithmicIdentityBinary40('data/alg'),
           'addition': AlgorithmicAdditionDecimal40('data/alg'),
           'multiplication': AlgorithmicMultiplicationDecimal40('data/alg'),
           'sort': AlgorithmicSortProblem('data/alg'),
           'reverse': AlgorithmicReverseProblem('data/alg'),
           'arithmatic': Arithmatic('data/arithmatic'),
           'sst': SST(data_path="data/sst/",
                 add_subtrees=True,
                 pretrained=True,
                 pretrained_path="data/sst/filtered_glove.txt",
                 embedding_size=300)}

  hparams.vocab_size = tasks[hparams.task_name].vocab_length
  hparams.output_dim = len(tasks[hparams.task_name].target_vocab)
  transformer_params = transformer_small_hparams(vocab_size=hparams.vocab_size,
                                                 output_dim=hparams.output_dim,
                                                 input_dim=hparams.input_dim,
                                                 encoder_attention_dir=hparams.encoder_attention_dir)

  lstm_params = lstm_small_hparams(vocab_size=hparams.vocab_size,
                                          output_dim=hparams.output_dim,
                                          input_dim=hparams.input_dim)



  model_params = {"transformer": transformer_params,
                  "utransformer": transformer_params,
                  "lstm": lstm_params,
                  "bilstm": lstm_params}


  if hparams.save_dir is None:
    hparams.save_dir = os.path.join(hparams.log_dir,hparams.task_name,
                                    '_'.join([hparams.model,
                                              'depth'+str(model_params[hparams.model].depth),
                                              'hidden_dim'+str(model_params[hparams.model].hidden_dim),
                                              'batch_size'+str(model_params[hparams.model].batch_size),
                                              hparams.exp_name]))

  model = Models[hparams.model](model_params[hparams.model],
                                task= tasks[hparams.task_name],
                                scope=hparams.model)

  trainer = Seq2SeqTrainer(hparams, model, tasks[hparams.task_name])
  trainer.train()