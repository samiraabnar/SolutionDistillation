
import tensorflow as tf

from distill.algorithmic_piplines import AlgorithmicTrainer
from distill.data_util.prep_algorithmic import AlgorithmicIdentityBinary40, AlgorithmicAdditionBinary40, \
  AlgorithmicMultiplicationBinary40, AlgorithmicSortProblem, AlgorithmicReverseProblem
from distill.layers.tree_lstm import TreeLSTM
from distill.models.lstm_seq2seq import LSTMSeq2Seq, BidiLSTMSeq2Seq
from distill.models.sentiment_tree_lstm import SentimentTreeLSTM
from distill.models.sentiment_lstm import SentimentLSTM
from distill.layers.lstm import LSTM
from distill.layers.bilstm import BiLSTM

import os

from distill.models.transformer import Transformer
from distill.pipelines import SSTRepDistiller

tf.logging.set_verbosity(tf.logging.INFO)

tf.app.flags.DEFINE_string("exp_name", "trial", "")
tf.app.flags.DEFINE_string("task_name", "identity", "identity | addition| multiplication | sort | reverse")
tf.app.flags.DEFINE_string("log_dir", "logs", "")
tf.app.flags.DEFINE_string("save_dir", None, "")
tf.app.flags.DEFINE_string("model", "transformer", "transformer | lstm | bilstm")


tf.app.flags.DEFINE_integer("hidden_dim", 128, "")
tf.app.flags.DEFINE_integer("depth", 1, "")
tf.app.flags.DEFINE_integer("input_dim", None, "")
tf.app.flags.DEFINE_integer("output_dim", 1, "")
tf.app.flags.DEFINE_integer("number_of_heads", 4, "")
tf.app.flags.DEFINE_integer("ff_filter_size", 5, "")
tf.app.flags.DEFINE_float("initializer_gain", 0.1, "")
tf.app.flags.DEFINE_float("label_smoothing", 0.1, "")


tf.app.flags.DEFINE_float("input_dropout_keep_prob", 0.75, "")
tf.app.flags.DEFINE_float("hidden_dropout_keep_prob", 0.5, "")

tf.app.flags.DEFINE_float("learning_rate", 0.00001, "")
tf.app.flags.DEFINE_float("l2_rate", 0.0005, "")


tf.app.flags.DEFINE_integer("batch_size", 32, "")
tf.app.flags.DEFINE_integer("training_iterations", 30000, "")

tf.app.flags.DEFINE_integer("vocab_size", 3, "")
tf.app.flags.DEFINE_integer("embedding_dim", 300, "embeddings dim")


tf.app.flags.DEFINE_string("pretrained_embedding_path", "data/sst/filtered_glove.txt", "pretrained embedding path")
tf.app.flags.DEFINE_string("data_path", "./data", "data path")


hparams = tf.app.flags.FLAGS


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
    self.attention_mechanism = attention_mechanism


if __name__ == '__main__':
  if hparams.save_dir is None:
    hparams.save_dir = os.path.join(hparams.log_dir,hparams.task_name,
                                    '_'.join([hparams.model,
                                              'depth'+str(hparams.depth),
                                              'hidden_dim'+str(hparams.hidden_dim),
                                              hparams.exp_name]))

  Models = {"lstm": LSTMSeq2Seq,
            "lstm_bidi": BidiLSTMSeq2Seq,
            "transformer": Transformer}


  tasks = {'identity': AlgorithmicIdentityBinary40('data/alg'),
           'addition': AlgorithmicAdditionBinary40('data/alg'),
           'multiplication': AlgorithmicMultiplicationBinary40('data/alg'),
           'sort': AlgorithmicSortProblem('data/alg'),
           'reverse': AlgorithmicReverseProblem('data/alg')}

  hparams.vocab_size = tasks[hparams.task_name].num_symbols + 1

  transformer_params = TransformerHparam(input_dim=hparams.input_dim,
                          hidden_dim=hparams.hidden_dim,
                          output_dim=hparams.output_dim,
                          depth=hparams.depth,
                          number_of_heads=hparams.number_of_heads,
                          ff_filter_size=hparams.ff_filter_size,
                          initializer_gain=hparams.initializer_gain,
                          batch_size=hparams.batch_size,
                          pretrained_embedding_path=hparams.pretrained_embedding_path,
                          input_dropout_keep_prob=hparams.input_dropout_keep_prob,
                          hidden_dropout_keep_prob=hparams.hidden_dropout_keep_prob,
                          vocab_size=hparams.vocab_size,
                          label_smoothing=hparams.label_smoothing
                          )
  lstm_params = LSTMHparam(input_dim=hparams.input_dim,
                                         hidden_dim=hparams.hidden_dim,
                                         output_dim=hparams.output_dim,
                                         depth=hparams.depth,
                                         number_of_heads=hparams.number_of_heads,
                                         ff_filter_size=hparams.ff_filter_size,
                                         initializer_gain=hparams.initializer_gain,
                                         batch_size=hparams.batch_size,
                                         pretrained_embedding_path=hparams.pretrained_embedding_path,
                                         input_dropout_keep_prob=hparams.input_dropout_keep_prob,
                                         hidden_dropout_keep_prob=hparams.hidden_dropout_keep_prob,
                                         vocab_size=hparams.vocab_size,
                                         label_smoothing=hparams.label_smoothing,
                                         attention_mechanism="final"
                                         )


  model = Models[hparams.model](model_params, scope=hparams.model)
  trainer = AlgorithmicTrainer(hparams, model, tasks[hparams.task_name])
  trainer.train()