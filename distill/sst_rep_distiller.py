import tensorflow as tf
import numpy as np
from distill.data_util.prep_sst import SST
from distill.data_util.vocab import PretrainedVocab
from distill.layers.tree_lstm import TreeLSTM
from distill.models.sentiment_tree_lstm import SentimentTreeLSTM
from distill.models.sentiment_lstm import SentimentLSTM
from distill.layers.lstm import LSTM
from distill.layers.bilstm import BiLSTM

from distill.common.distill_util import get_logit_distill_loss
import os

from distill.sst_distiller import SSTDistiller

tf.logging.set_verbosity(tf.logging.INFO)

tf.app.flags.DEFINE_string("exp_name", "trial", "")
tf.app.flags.DEFINE_string("task_name", "sst_distill", "")
tf.app.flags.DEFINE_string("log_dir", "logs", "")
tf.app.flags.DEFINE_string("save_dir", None, "")

tf.app.flags.DEFINE_string("teacher_model_type", "bidi", "")
tf.app.flags.DEFINE_string("student_model_type", "plain", "")
tf.app.flags.DEFINE_boolean("pretrain_teacher", True, "")
tf.app.flags.DEFINE_integer("teacher_pretraining_iters", 100, "")

tf.app.flags.DEFINE_string("model_type", "bidi_to_plain", "")
tf.app.flags.DEFINE_integer("hidden_dim", 50, "")
tf.app.flags.DEFINE_integer("depth", 1, "")
tf.app.flags.DEFINE_integer("input_dim", None, "")
tf.app.flags.DEFINE_integer("output_dim", 2, "")
tf.app.flags.DEFINE_string("attention_mechanism", None, "")

tf.app.flags.DEFINE_string("loss_type", "root_loss", "")
tf.app.flags.DEFINE_float("input_dropout_keep_prob", 0.75, "")
tf.app.flags.DEFINE_float("hidden_dropout_keep_prob", 0.5, "")

tf.app.flags.DEFINE_float("learning_rate", 0.00001, "")
tf.app.flags.DEFINE_float("l2_rate", 0.00001, "")

tf.app.flags.DEFINE_integer("batch_size", 32, "")
tf.app.flags.DEFINE_integer("training_iterations", 30000, "")

tf.app.flags.DEFINE_integer("vocab_size", 8000, "")
tf.app.flags.DEFINE_integer("embedding_dim", 100, "embeddings dim")


tf.app.flags.DEFINE_string("pretrained_embedding_path", "/Users/samiraabnar/Codes/Data/word_embeddings/glove.6B/glove.6B.100d.txt", "pretrained embedding path")
tf.app.flags.DEFINE_string("data_path", "./data", "data path")


hparams = tf.app.flags.FLAGS


class SSTRepDistiller(SSTDistiller):
  def __init__(self, config, student_model, teacher_model):
    super(self, SSTRepDistiller)

  def get_train_op(self, loss, params, scope=""):
    # add training op
    with tf.variable_scope(scope):
      self.global_step = tf.train.get_or_create_global_step()

      loss_l2 = tf.add_n([tf.nn.l2_loss(p) for p in params]) * self.config.l2_rate

      loss += loss_l2

      starter_learning_rate = 0.0001
      learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,
                                                 1000, 0.96, staircase=True)
      opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
      grads_and_vars = opt.compute_gradients(loss, params)
      gradients, variables = zip(*grads_and_vars)
      self.gradient_norm = tf.global_norm(gradients)
      clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5)
      self.param_norm = tf.global_norm(params)

      # Include batch norm mean and variance in gradient descent updates
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(update_ops):
        # Fetch self.updates to apply gradients to all trainable parameters.
        updates = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)

    return updates, learning_rate


  def get_data_itaratoes(self):
    dataset = tf.data.TFRecordDataset(SST.get_tfrecord_path("data/sst", mode="train"))
    dataset = dataset.map(SST.parse_full_sst_tree_examples)
    dataset = dataset.padded_batch(self.config.batch_size, padded_shapes=SST.get_padded_shapes(), drop_remainder=True)
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.repeat()
    iterator = dataset.make_initializable_iterator()

    dev_dataset = tf.data.TFRecordDataset(SST.get_tfrecord_path("data/sst", mode="dev"))
    dev_dataset = dev_dataset.map(SST.parse_full_sst_tree_examples)
    dev_dataset = dev_dataset.shuffle(buffer_size=1101)
    dev_dataset = dev_dataset.repeat()
    dev_dataset = dev_dataset.padded_batch(1101, padded_shapes=SST.get_padded_shapes(),
                                           drop_remainder=True)
    dev_iterator = dev_dataset.make_initializable_iterator()

    test_dataset = tf.data.TFRecordDataset(SST.get_tfrecord_path("data/sst", mode="test"))
    test_dataset = test_dataset.map(SST.parse_full_sst_tree_examples)
    test_dataset = test_dataset.shuffle(buffer_size=2210)
    test_dataset = test_dataset.repeat()
    test_dataset = test_dataset.padded_batch(2210, padded_shapes=SST.get_padded_shapes(),
                                             drop_remainder=True)
    test_iterator = test_dataset.make_initializable_iterator()

    return iterator, dev_iterator, test_iterator

  def build_train_graph(self):
    self.student.build_graph(self.pretrained_word_embeddings)
    self.teacher.build_graph(self.pretrained_word_embeddings)

    train_iterator, dev_iterator, test_iterator = self.get_data_itaratoes()

    train_examples = train_iterator.get_next()
    dev_examples = dev_iterator.get_next()
    test_examples =  test_iterator.get_next()

    student_train_output_dic = self.student.apply(train_examples)
    teacher_train_output_dic = self.teacher.apply(train_examples)

    student_dev_output_dic = self.student.apply(dev_examples)
    teacher_dev_output_dic = self.teacher.apply(dev_examples)

    student_test_output_dic = self.student.apply(test_examples)
    teacher_test_output_dic = self.teacher.apply(test_examples)

    tf.summary.scalar("loss", student_train_output_dic[self.config.loss_type], family="student_train")
    tf.summary.scalar("accuracy", student_train_output_dic["root_accuracy"], family="student_train")

    tf.summary.scalar("loss", student_dev_output_dic[self.config.loss_type], family="student_dev")
    tf.summary.scalar("accuracy", student_dev_output_dic["root_accuracy"], family="student_dev")

    tf.summary.scalar("loss", student_test_output_dic[self.config.loss_type], family="student_test")
    tf.summary.scalar("accuracy", student_test_output_dic["root_accuracy"], family="student_test")

    tf.summary.scalar("loss", teacher_train_output_dic[self.config.loss_type], family="teacher_train")
    tf.summary.scalar("accuracy", teacher_train_output_dic["root_accuracy"], family="teacher_train")

    tf.summary.scalar("loss", teacher_dev_output_dic[self.config.loss_type], family="teacher_dev")
    tf.summary.scalar("accuracy", teacher_dev_output_dic["root_accuracy"], family="teacher_dev")

    tf.summary.scalar("loss", teacher_test_output_dic[self.config.loss_type], family="teacher_test")
    tf.summary.scalar("accuracy", teacher_test_output_dic["root_accuracy"], family="teacher_test")


    update_op, learning_rate = self.get_train_op(student_train_output_dic[self.config.loss_type],
                                  student_train_output_dic["trainable_vars"],
                                                 scope="main")

    distill_loss = get_logit_distill_loss(student_train_output_dic['logits'],teacher_train_output_dic['logits'])
    tf.summary.scalar("distill loss", distill_loss, family="student_train")

    distill_op, learning_rate = self.get_train_op(distill_loss, student_train_output_dic["trainable_vars"],
                                                  scope="distill")



    scaffold = tf.train.Scaffold(local_init_op=tf.group(tf.local_variables_initializer(),
                                                        train_iterator.initializer,
                                                        dev_iterator.initializer,
                                                        test_iterator.initializer))

    return update_op,distill_op, scaffold


  def train(self):
    update_op, distill_op, scaffold  = self.build_train_graph()
    with tf.train.MonitoredTrainingSession(checkpoint_dir=self.config.save_dir, scaffold=scaffold) as sess:
      for _ in np.arange(self.config.training_iterations):
        sess.run(update_op)
        #sess.run(distill_op)


if __name__ == '__main__':
  if hparams.save_dir is None:
    hparams.save_dir = os.path.join(hparams.log_dir,hparams.task_name, '_'.join([hparams.model_type, hparams.loss_type,'depth'+str(hparams.depth),'hidden_dim'+str(hparams.hidden_dim),hparams.exp_name]))

  Models = {"plain": LSTM,
            "bidi": BiLSTM,
            "tree": TreeLSTM}

  student = SentimentLSTM(hparams, model=LSTM, scope="student")
  teacher = SentimentLSTM(hparams, model=BiLSTM, scope="teacher")

  trainer = SSTDistiller(config=hparams, student_model=student, teacher_model=teacher)
  trainer.train()