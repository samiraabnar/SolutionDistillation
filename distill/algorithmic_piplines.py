import tensorflow as tf
import numpy as np

from distill.basic_trainer import Trainer
from distill.common.metrics import padded_cross_entropy_loss, get_eval_metrics


class AlgorithmicTrainer(Trainer):
  def __init__(self, config, model_obj, task):
    super(AlgorithmicTrainer, self).__init__(config, model_obj)
    self.task = task


  def get_data_itaratoes(self):
    dataset = tf.data.TFRecordDataset(self.task.get_tfrecord_path(mode="train"))
    dataset = dataset.map(self.task.parse_examples)
    dataset = dataset.padded_batch(self.config.batch_size, padded_shapes=self.task.get_padded_shapes())
    train_iterator = dataset.make_initializable_iterator()

    dataset = tf.data.TFRecordDataset(self.task.get_tfrecord_path(mode="dev"))
    dataset = dataset.map(self.task.parse_examples)
    dataset = dataset.padded_batch(self.config.batch_size, padded_shapes=self.task.get_padded_shapes())
    dev_iterator = dataset.make_initializable_iterator()

    dataset = tf.data.TFRecordDataset(self.task.get_tfrecord_path(mode="test"))
    dataset = dataset.map(self.task.parse_examples)
    dataset = dataset.padded_batch(self.config.batch_size, padded_shapes=self.task.get_padded_shapes())
    test_iterator = dataset.make_initializable_iterator()

    return train_iterator, dev_iterator, test_iterator


  def compute_loss(self,logits, targets):
    xentropy, weights = padded_cross_entropy_loss(
      logits, targets, self.config.label_smoothing, self.config.vocab_size)
    loss = tf.reduce_sum(xentropy) / tf.reduce_sum(weights)

    return loss


  def add_metric_summaries(self, logits, labels, family):
    eval_metrics = get_eval_metrics(logits, labels, self.model.hparams)
    for metric in eval_metrics:
      tf.logging.info(metric)
      tf.logging.info(eval_metrics[metric])
      tf.summary.scalar(metric, tf.reduce_mean(eval_metrics[metric]), family=family)



  def build_train_graph(self):
    train_iterator, dev_iterator, test_iterator = self.get_data_itaratoes()

    train_examples = train_iterator.get_next()
    dev_examples = dev_iterator.get_next()
    test_examples = test_iterator.get_next()



    self.model.create_vars(reuse=False)

    train_output_dic = self.model.apply(train_examples, is_train=True)
    dev_output_dic = self.model.apply(dev_examples, is_train=False)
    test_output_dic = self.model.apply(test_examples, is_train=False)

    train_loss = self.compute_loss(train_output_dic['logits'], train_output_dic['targets'])
    dev_loss = self.compute_loss(dev_output_dic['logits'], dev_output_dic['targets'])
    test_loss = self.compute_loss(test_output_dic['logits'], test_output_dic['targets'])

    tf.summary.scalar("loss", train_loss, family="train")
    tf.summary.scalar("loss", dev_loss, family="dev")
    tf.summary.scalar("loss", test_loss, family="test")

    self.add_metric_summaries(train_output_dic['logits'], train_output_dic['targets'], "train")
    self.add_metric_summaries(dev_output_dic['logits'], dev_output_dic['targets'], "train")
    self.add_metric_summaries(test_output_dic['logits'], test_output_dic['targets'], "train")



    update_op, learning_rate = self.get_train_op(train_loss, train_output_dic["trainable_vars"],
                                                 start_learning_rate=0.0005,
                                                 base_learning_rate=0.001, warmup_steps=1000
                                                 )
    tf.summary.scalar("learning_rate", learning_rate, family="train")

    scaffold = tf.train.Scaffold(local_init_op=tf.group(tf.local_variables_initializer(),
                                                        train_iterator.initializer,
                                                        dev_iterator.initializer,
                                                        test_iterator.initializer),
                                 init_feed_dict={})

    return update_op, scaffold, train_output_dic, dev_output_dic, test_output_dic
