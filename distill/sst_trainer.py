import tensorflow as tf
import numpy as np
from distill.data_util.prep_sst import SST
from distill.models.sentiment_tree_lstm import SentimentTreeLSTM
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



class SSTTrainer(object):
  def __init__(self, config):
    self.config = config
    self.sst = SST("data/sst")
    config.vocab_size = len(self.sst.vocab)
    self.sentimen_tree_lstm = SentimentTreeLSTM(self.config)


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
    self.sentimen_tree_lstm.build_graph()

    train_iterator, dev_iterator = self.get_data_itaratoes()
    train_output_dic = self.sentimen_tree_lstm.apply(train_iterator.get_next())
    tf.summary.scalar("loss", train_output_dic[self.config.loss_type], family="train")
    tf.summary.scalar("accuracy", train_output_dic["root_accuracy"], family="train")

    dev_output_dic = self.sentimen_tree_lstm.apply(dev_iterator.get_next())
    tf.summary.scalar("loss", dev_output_dic[self.config.loss_type], family="dev")
    tf.summary.scalar("accuracy", dev_output_dic["root_accuracy"], family="dev")


    update_op = self.get_train_op(train_output_dic[self.config.loss_type],train_output_dic["trainable_vars"])

    scaffold = tf.train.Scaffold(local_init_op=tf.group(tf.local_variables_initializer(),
                                                        train_iterator.initializer,
                                                        dev_iterator.initializer))

    return update_op, scaffold


  def train(self):
    update_op, scaffold  = self.build_train_graph()
    with tf.train.MonitoredTrainingSession(checkpoint_dir=self.config.save_dir, scaffold=scaffold) as sess:
      for _ in np.arange(self.config.max_iterations):
        sess.run(update_op)



if __name__ == '__main__':
  trainer = SSTTrainer(config=Config)
  trainer.train()