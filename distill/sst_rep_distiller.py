import tensorflow as tf
from distill.layers.tree_lstm import TreeLSTM
from distill.models.sentiment_tree_lstm import SentimentTreeLSTM
from distill.models.sentiment_lstm import SentimentLSTM
from distill.layers.lstm import LSTM
from distill.layers.bilstm import BiLSTM

import os

from distill.pipelines import SSTRepDistiller

tf.logging.set_verbosity(tf.logging.INFO)

tf.app.flags.DEFINE_string("exp_name", "trial", "")
tf.app.flags.DEFINE_string("task_name", "sst_distill", "")
tf.app.flags.DEFINE_string("log_dir", "logs", "")
tf.app.flags.DEFINE_string("save_dir", None, "")

tf.app.flags.DEFINE_string("teacher_model", "bidi", "")
tf.app.flags.DEFINE_string("student_model", "plain", "")
tf.app.flags.DEFINE_boolean("pretrain_teacher", True, "")
tf.app.flags.DEFINE_integer("teacher_pretraining_iters", 100, "")
tf.app.flags.DEFINE_string("rep_loss_mode", 'softmax_cross_ent', "representation loss type (squared,softmax_cross_ent,sigmoid_cross_ent")


tf.app.flags.DEFINE_string("model_type", "rep_bidi_to_plain", "")
tf.app.flags.DEFINE_integer("hidden_dim", 64, "")
tf.app.flags.DEFINE_integer("depth", 1, "")
tf.app.flags.DEFINE_integer("input_dim", None, "")
tf.app.flags.DEFINE_integer("output_dim", 1, "")
tf.app.flags.DEFINE_string("attention_mechanism", None, "")

tf.app.flags.DEFINE_string("loss_type", "root_loss", "")
tf.app.flags.DEFINE_float("input_dropout_keep_prob", 0.75, "")
tf.app.flags.DEFINE_float("hidden_dropout_keep_prob", 0.5, "")

tf.app.flags.DEFINE_float("learning_rate", 0.00001, "")
tf.app.flags.DEFINE_float("l2_rate", 0.0005, "")

tf.app.flags.DEFINE_integer("batch_size", 32, "")
tf.app.flags.DEFINE_integer("training_iterations", 30000, "")

tf.app.flags.DEFINE_integer("vocab_size", 8000, "")
tf.app.flags.DEFINE_integer("embedding_dim", 300, "embeddings dim")


tf.app.flags.DEFINE_string("pretrained_embedding_path", "data/sst/filtered_glove.txt", "pretrained embedding path")
tf.app.flags.DEFINE_string("data_path", "./data", "data path")


hparams = tf.app.flags.FLAGS


if __name__ == '__main__':
  if hparams.save_dir is None:
    hparams.save_dir = os.path.join(hparams.log_dir,hparams.task_name, '_'.join([hparams.rep_loss_mode, hparams.model_type, hparams.loss_type,'depth'+str(hparams.depth),'hidden_dim'+str(hparams.hidden_dim),hparams.exp_name]))

  Models = {"plain": LSTM,
            "bidi": BiLSTM,
            "tree": TreeLSTM}

  student = SentimentLSTM(hparams, model=Models[hparams.student_model], scope="student")
  if hparams.teacher_model == "tree":
    teacher = SentimentTreeLSTM(hparams, model=Models[hparams.teacher_model], scope="teacher")
  else:
    teacher = SentimentLSTM(hparams, model=Models[hparams.teacher_model], scope="teacher")

  trainer = SSTRepDistiller(config=hparams, student_model=student, teacher_model=teacher)
  trainer.train()