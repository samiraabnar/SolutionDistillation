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
tf.app.flags.DEFINE_boolean("train_teacher", True, "")
tf.app.flags.DEFINE_boolean("train_student", False, "")
tf.app.flags.DEFINE_boolean("distill_rep", False, "")
tf.app.flags.DEFINE_boolean("distill_logit", True, "")

tf.app.flags.DEFINE_string("student_model", "plain", "")
tf.app.flags.DEFINE_boolean("pretrain_teacher", True, "")
tf.app.flags.DEFINE_integer("teacher_pretraining_iters", 100, "")
tf.app.flags.DEFINE_string("rep_loss_mode", 'squared', "representation loss type (squared,softmax_cross_ent,sigmoid_cross_ent")


tf.app.flags.DEFINE_string("model_type", "rep_bidi_to_plain", "")
tf.app.flags.DEFINE_integer("student_hidden_dim", 128, "")
tf.app.flags.DEFINE_integer("student_depth", 2, "")
tf.app.flags.DEFINE_integer("teacher_hidden_dim", 168, "")
tf.app.flags.DEFINE_integer("teacher_depth", 2, "")
tf.app.flags.DEFINE_integer("input_dim", None, "")
tf.app.flags.DEFINE_integer("output_dim", 1, "")
tf.app.flags.DEFINE_string("student_attention_mechanism", None, "")
tf.app.flags.DEFINE_string("teacher_attention_mechanism", None, "")
tf.app.flags.DEFINE_string("sent_rep_mode", 'final', "")


tf.app.flags.DEFINE_string("loss_type", "root_loss", "")
tf.app.flags.DEFINE_float("input_dropout_keep_prob", 0.75, "")
tf.app.flags.DEFINE_float("hidden_dropout_keep_prob", 0.5, "")

tf.app.flags.DEFINE_float("learning_rate", 0.00001, "")
tf.app.flags.DEFINE_float("l2_rate", 0.001, "")

tf.app.flags.DEFINE_integer("batch_size", 32, "")
tf.app.flags.DEFINE_integer("training_iterations", 30000, "")

tf.app.flags.DEFINE_integer("vocab_size", 8000, "")
tf.app.flags.DEFINE_integer("embedding_dim", 300, "embeddings dim")


tf.app.flags.DEFINE_string("pretrained_embedding_path", "data/sst/filtered_glove.txt", "pretrained embedding path")
tf.app.flags.DEFINE_string("data_path", "./data", "data path")


hparams = tf.app.flags.FLAGS


class Hparam(object):
  def __init__(self, input_dim,
               hidden_dim,
               output_dim,
               depth,
               attention_mechanism,
               batch_size,
               pretrained_embedding_path,
               input_dropout_keep_prob,
               hidden_dropout_keep_prob,
               loss_type,
               sent_rep_mode,
               embedding_dim):
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.output_dim = output_dim
    self.depth = depth
    self.attention_mechanism = attention_mechanism
    self.batch_size = batch_size
    self.pretrained_embedding_path = pretrained_embedding_path
    self.input_dropout_keep_prob = input_dropout_keep_prob
    self.hidden_dropout_keep_prob = hidden_dropout_keep_prob
    self.loss_type = loss_type
    self.sent_rep_mode = sent_rep_mode
    self.vocab_size = None
    self.embedding_dim = embedding_dim

if __name__ == '__main__':
  if hparams.save_dir is None:
    hparams.save_dir = os.path.join(hparams.log_dir,hparams.task_name, '_'.join([hparams.rep_loss_mode, hparams.model_type, hparams.loss_type,'std_depth'+str(hparams.student_depth),'teacher_depth'+str(hparams.teacher_depth),'std_hidden_dim'+str(hparams.student_hidden_dim),'teacher_hidden_dim'+str(hparams.teacher_hidden_dim),hparams.exp_name]))

  Models = {"plain": LSTM,
            "bidi": BiLSTM,
            "tree": TreeLSTM}


  student_params = Hparam(input_dim=hparams.input_dim,
                          hidden_dim=hparams.student_hidden_dim,
                          output_dim=hparams.output_dim,
                          depth=hparams.student_depth,
                          attention_mechanism=hparams.student_attention_mechanism,
                          batch_size=hparams.batch_size,
                          pretrained_embedding_path=hparams.pretrained_embedding_path,
                          input_dropout_keep_prob=hparams.input_dropout_keep_prob,
                          hidden_dropout_keep_prob=hparams.hidden_dropout_keep_prob,
                          loss_type=hparams.loss_type,
                          sent_rep_mode=hparams.sent_rep_mode,
                          embedding_dim=hparams.embedding_dim)

  teacher_params = Hparam(input_dim=hparams.input_dim,
                          hidden_dim=hparams.teacher_hidden_dim,
                          output_dim=hparams.output_dim,
                          depth=hparams.teacher_depth,
                          attention_mechanism=hparams.teacher_attention_mechanism,
                          batch_size=hparams.batch_size,
                          pretrained_embedding_path=hparams.pretrained_embedding_path,
                          input_dropout_keep_prob=hparams.input_dropout_keep_prob,
                          hidden_dropout_keep_prob=hparams.hidden_dropout_keep_prob,
                          loss_type=hparams.loss_type,
                          sent_rep_mode=hparams.sent_rep_mode,
                          embedding_dim=hparams.embedding_dim)


  student = SentimentLSTM(student_params, model=Models[hparams.student_model], scope="student")
  if hparams.teacher_model == "tree":
    teacher = SentimentTreeLSTM(teacher_params, model=Models[hparams.teacher_model], scope="teacher")
  else:
    teacher = SentimentLSTM(teacher_params, model=Models[hparams.teacher_model], scope="teacher")

  trainer = SSTRepDistiller(config=hparams, student_model=student, teacher_model=teacher)
  trainer.train()