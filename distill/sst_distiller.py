import tensorflow as tf
import numpy as np
from distill.data_util.prep_sst import SST
from distill.models.sentiment_tree_lstm import SentimentTreeLSTM
from distill.models.sentiment_lstm import SentimentLSTM

from distill.common.distill_util import get_logit_distill_loss
import os

tf.logging.set_verbosity(tf.logging.INFO)

class Config(object):
  """Holds model hyperparams and data information.
  Model objects are passed a Config() object at instantiation.
  """
  embed_size = 35
  label_size = 2
  early_stopping = 2
  anneal_threshold = 0.99
  anneal_by = 1.5
  max_iterations = 3000000
  lr = 0.001
  l2 = 0.001
  vocab_size = 80000
  batch_size = 10
  loss_type = 'root_loss'
  log_dir = 'logs'
  task_name = 'sst'
  model_name = 'tree_lstm'

  save_dir = os.path.join(log_dir, task_name, model_name+"_"+loss_type+"_lr=%d_l2=%f_lr=%f" % (embed_size, l2, lr))



class SSTDistiller(object):
  def __init__(self, config, student_model_class, teacher_model_class):
    self.config = config
    self.sst = SST("data/sst")
    config.vocab_size = len(self.sst.vocab)
    self.student = student_model_class(self.config)
    self.teacher = teacher_model_class(self.config)

  def get_train_op(self, loss, params):
    # add training op
    self.global_step = tf.train.get_or_create_global_step()

    # Learning rate is linear from step 0 to self.FLAGS.lr_warmup. Then it decays as 1/sqrt(timestep).
    opt = tf.train.AdamOptimizer(learning_rate=self.config.lr)
    grads_and_vars = opt.compute_gradients(loss, params)
    gradients, variables = zip(*grads_and_vars)
    self.gradient_norm = tf.global_norm(gradients)
    clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
    self.param_norm = tf.global_norm(params)

    # Include batch norm mean and variance in gradient descent updates
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      # Fetch self.updates to apply gradients to all trainable parameters.
      updates = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)

    return updates


  def get_data_itaratoes(self):
    dataset = tf.data.TFRecordDataset(SST.get_tfrecord_path("data/sst", mode="train"))
    dataset = dataset.map(SST.parse_sst_tree_examples)
    dataset = dataset.padded_batch(self.config.batch_size, padded_shapes=SST.get_padded_shapes(), drop_remainder=True)
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.repeat()
    iterator = dataset.make_initializable_iterator()

    dev_dataset = tf.data.TFRecordDataset(SST.get_tfrecord_path("data/sst", mode="train"))
    dev_dataset = dev_dataset.map(SST.parse_sst_tree_examples)
    dev_dataset = dev_dataset.shuffle(buffer_size=1101)
    dev_dataset = dev_dataset.repeat()
    dev_dataset = dev_dataset.padded_batch(1101, padded_shapes=SST.get_padded_shapes(),
                                           drop_remainder=True)
    dev_iterator = dev_dataset.make_initializable_iterator()

    return iterator, dev_iterator

  def build_train_graph(self):
    self.student.build_graph()
    self.teacher.build_graph()


    train_iterator, dev_iterator = self.get_data_itaratoes()


    student_train_output_dic = self.student.apply(train_iterator.get_next())
    teacher_train_output_dic = self.teacher.apply(train_iterator.get_next())
    student_dev_output_dic = self.student.apply(dev_iterator.get_next())


    tf.summary.scalar("loss", student_train_output_dic[self.config.loss_type], family="train")
    tf.summary.scalar("accuracy", student_train_output_dic["root_accuracy"], family="train")
    tf.summary.scalar("loss", student_dev_output_dic[self.config.loss_type], family="dev")
    tf.summary.scalar("accuracy", student_dev_output_dic["root_accuracy"], family="dev")


    update_op = self.get_train_op(student_train_output_dic[self.config.loss_type],
                                  student_train_output_dic["trainable_vars"])

    distill_loss = get_logit_distill_loss(student_train_output_dic['logits'],teacher_train_output_dic['logits'])
    distill_op = self.get_train_op(distill_loss, student_train_output_dic["trainable_vars"])



    scaffold = tf.train.Scaffold(local_init_op=tf.group(tf.local_variables_initializer(),
                                                        train_iterator.initializer,
                                                        dev_iterator.initializer))

    return update_op,distill_op, scaffold


  def train(self):
    update_op, distill_op, scaffold  = self.build_train_graph()
    with tf.train.MonitoredTrainingSession(checkpoint_dir=self.config.save_dir, scaffold=scaffold) as sess:
      for _ in np.arange(self.config.max_iterations):
        sess.run(update_op)
        sess.run(distill_op)






if __name__ == '__main__':
  trainer = SSTDistiller(config=Config, student_model_class=SentimentLSTM, teacher_model_class=SentimentTreeLSTM)
  trainer.train()