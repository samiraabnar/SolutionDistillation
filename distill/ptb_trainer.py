import tensorflow as tf
import numpy as np

from distill.common.hparams import TransformerHparam, LSTMHparam
from distill.data_util.prep_ptb import PTB
from distill.data_util.prep_sentwiki import SentWiki
from distill.models.lm_lstm import LmLSTM
import os

from distill.pipelines.lm import LMTrainer

tf.logging.set_verbosity(tf.logging.INFO)


tf.app.flags.DEFINE_string("exp_name", "trial", "")
tf.app.flags.DEFINE_string("task_name", "sent_wiki", "ptb_lm")
tf.app.flags.DEFINE_string("log_dir", "logs", "")
tf.app.flags.DEFINE_string("save_dir", None, "")

tf.app.flags.DEFINE_string("model", "lm_lstm", "")
tf.app.flags.DEFINE_string("encoder_attention_dir", "top_down", "top_down | bottom_up")
tf.app.flags.DEFINE_integer("hidden_dim", 512, "")
tf.app.flags.DEFINE_integer("output_dim", 8000, "")
tf.app.flags.DEFINE_integer("input_dim", 8000, "")
tf.app.flags.DEFINE_integer("number_of_heads", 4, "")
tf.app.flags.DEFINE_integer("ff_filter_size", 512, "")
tf.app.flags.DEFINE_float("initializer_gain", 1.0, "")
tf.app.flags.DEFINE_float("label_smoothing", 0.1, "")
tf.app.flags.DEFINE_boolean('train_embeddings', True, " False | True")
tf.app.flags.DEFINE_string('sent_rep_mode', "final", "none | final | all")
tf.app.flags.DEFINE_string('attention_mechanism',None, 'attention_mechanism')
tf.app.flags.DEFINE_integer("depth", 3, "")


tf.app.flags.DEFINE_float("input_dropout_keep_prob", 0.75, "")
tf.app.flags.DEFINE_float("hidden_dropout_keep_prob", 0.5, "")

tf.app.flags.DEFINE_float("learning_rate", 0.0001, "")
tf.app.flags.DEFINE_boolean("decay_learning_rate", True, "True | False")
tf.app.flags.DEFINE_float("l2_rate", 0.0001, "")

tf.app.flags.DEFINE_integer("batch_size", 32, "")
tf.app.flags.DEFINE_integer("training_iterations", 300000, "")

tf.app.flags.DEFINE_integer("vocab_size", 8000, "")
tf.app.flags.DEFINE_integer("embedding_dim", 512, "embeddings dim")
tf.app.flags.DEFINE_boolean("bidirectional", False, "If the LSTM layer is bidirectional")


tf.app.flags.DEFINE_string("pretrained_embedding_path", "data/sst/filtered_glove.txt", "pretrained embedding path")
tf.app.flags.DEFINE_string("data_path", "./data", "data path")


hparams = tf.app.flags.FLAGS





if __name__ == '__main__':
  Models = {"lm_lstm": LmLSTM}
  tasks = {'ptb_lm': PTB('data/ptb'),
           'sent_wiki': SentWiki('data/sent_wiki')}

  hparams.vocab_size = tasks[hparams.task_name].vocab_length
  hparams.output_dim = len(tasks[hparams.task_name].target_vocab)

  transformer_params = TransformerHparam(input_dim=hparams.input_dim,
                                         hidden_dim=hparams.hidden_dim,
                                         output_dim=hparams.output_dim,
                                         encoder_depth=hparams.depth,
                                         decoder_depth=0,
                                         number_of_heads=hparams.number_of_heads,
                                         ff_filter_size=hparams.ff_filter_size,
                                         initializer_gain=hparams.initializer_gain,
                                         batch_size=hparams.batch_size,
                                         input_dropout_keep_prob=hparams.input_dropout_keep_prob,
                                         hidden_dropout_keep_prob=hparams.hidden_dropout_keep_prob,
                                         vocab_size=hparams.vocab_size,
                                         label_smoothing=hparams.label_smoothing,
                                         encoder_self_attention_dir=hparams.encoder_attention_dir,
                                         decoder_self_attention_dir="top_down",
                                         decoder_cross_attention_dir="top_down",
                                         train_embeddings=hparams.train_embeddings,
                                         learning_rate=hparams.learning_rate
                                         )

  lstm_params = LSTMHparam(input_dim=hparams.input_dim,
                           hidden_dim=hparams.hidden_dim,
                           output_dim=hparams.output_dim,
                           encoder_depth=hparams.depth,
                           decoder_depth=0,
                           number_of_heads=hparams.number_of_heads,
                           ff_filter_size=hparams.ff_filter_size,
                           initializer_gain=hparams.initializer_gain,
                           batch_size=hparams.batch_size,
                           input_dropout_keep_prob=hparams.input_dropout_keep_prob,
                           hidden_dropout_keep_prob=hparams.hidden_dropout_keep_prob,
                           vocab_size=hparams.vocab_size,
                           label_smoothing=hparams.label_smoothing,
                           attention_mechanism=None,
                           sent_rep_mode=hparams.sent_rep_mode,
                           embedding_dim=hparams.embedding_dim,
                           train_embeddings = hparams.train_embeddings,
                           learning_rate=hparams.learning_rate)


  model_params = {"lm_lstm": lstm_params}


  if hparams.save_dir is None:
    hparams.save_dir = os.path.join(hparams.log_dir,hparams.task_name,
                                    '_'.join([hparams.model,
                                              'depth'+str(model_params[hparams.model].encoder_depth),
                                              'hidden_dim'+str(model_params[hparams.model].hidden_dim),
                                              'batch_size'+str(model_params[hparams.model].batch_size),
                                              hparams.exp_name]))

  model = Models[hparams.model](model_params[hparams.model],
                                  task=tasks[hparams.task_name],
                                  scope=hparams.model)
    
  model = Models[hparams.model](model_params[hparams.model],
                                  task=tasks[hparams.task_name],
                                  scope=hparams.model)

  trainer = LMTrainer(hparams, model, tasks[hparams.task_name])
  trainer.train()