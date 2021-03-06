import tensorflow as tf
import numpy as np
from distill.common.metrics import padded_cross_entropy_loss, get_eval_metrics
from distill.pipelines.basic_trainer import Trainer



class LMTrainer(Trainer):
  def __init__(self, config, model_obj, task):
    super(LMTrainer, self).__init__(config, model_obj)
    self.task = task

  def get_data_itarators(self):

    bucket_boundaries = list(np.arange(10,self.task.max_length,10))
    dataset = tf.data.TFRecordDataset(self.task.get_tfrecord_path(mode="train"))
    dataset = dataset.map(self.task.parse_examples)
    dataset = dataset.apply(
      tf.data.experimental.bucket_by_sequence_length(element_length_func=lambda x1, x2, x3, x4: tf.size(x1),
                                                     bucket_batch_sizes=[self.config.batch_size]*(len(bucket_boundaries)+1),
                                                     bucket_boundaries=bucket_boundaries,
                                                     padded_shapes=self.task.get_padded_shapes()))
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.repeat()
    train_iterator = dataset.make_initializable_iterator()

    dataset = tf.data.TFRecordDataset(self.task.get_tfrecord_path(mode="dev"))
    dataset = dataset.map(self.task.parse_examples)
    dataset = dataset.apply(
      tf.data.experimental.bucket_by_sequence_length(element_length_func=lambda x1, x2, x3, x4: tf.size(x1),
                                                     bucket_batch_sizes=[self.config.batch_size] * (len(bucket_boundaries)+1),
                                                     bucket_boundaries=bucket_boundaries,
                                                     padded_shapes=self.task.get_padded_shapes()))
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.repeat()
    dev_iterator = dataset.make_initializable_iterator()

    dataset = tf.data.TFRecordDataset(self.task.get_tfrecord_path(mode="test"))
    dataset = dataset.map(self.task.parse_examples)
    dataset = dataset.apply(
      tf.data.experimental.bucket_by_sequence_length(element_length_func=lambda x1, x2, x3, x4: tf.size(x1),
                                                     bucket_batch_sizes=[self.config.batch_size] * (len(bucket_boundaries)+1),
                                                     bucket_boundaries=bucket_boundaries,
                                                     padded_shapes=self.task.get_padded_shapes()))
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.repeat()
    test_iterator = dataset.make_initializable_iterator()

    return train_iterator, dev_iterator, test_iterator

  def compute_loss(self,logits, targets, softmax_temperature=1.0):
    xentropy, weights = padded_cross_entropy_loss(
      logits, targets, self.model.hparams.label_smoothing, self.config.output_dim,
      softmax_temperature=softmax_temperature,  gaussian_noise=self.task.if_label_gaussian_noise, gaussian_noise_scale=self.task.guassian_noise_scale)

    loss = tf.reduce_sum(xentropy) / tf.reduce_sum(weights)

    return loss

  def add_metric_summaries(self, logits, labels, family):
    eval_metrics = get_eval_metrics(logits, labels, self.model.hparams)
    for metric in eval_metrics:
      tf.summary.scalar(metric, tf.reduce_mean(eval_metrics[metric]), family=family)

  def get_metric_summaries_as_dic(self, logits, labels):
    metric_summaries = {}
    eval_metrics = get_eval_metrics(logits, labels, self.model.hparams)
    for metric in eval_metrics:
      metric_summaries[metric] = tf.reduce_mean(eval_metrics[metric])

    return metric_summaries


  def build_train_graph(self):
    self.model.create_vars()

    train_iterator, dev_iterator, test_iterator = self.get_data_itarators()

    train_output_dic = self.model.apply(train_iterator.get_next())
    dev_output_dic = self.model.apply(dev_iterator.get_next(), is_train=False)
    test_output_dic = self.model.apply(test_iterator.get_next(), is_train=False)


    self.add_metric_summaries(train_output_dic['logits'], train_output_dic['targets'], "train")
    self.add_metric_summaries(dev_output_dic['logits'], dev_output_dic['targets'], "dev")
    self.add_metric_summaries(test_output_dic['logits'], test_output_dic['targets'], "test")


    train_loss = self.compute_loss(train_output_dic['logits'], train_output_dic['targets'])
    dev_loss = self.compute_loss(dev_output_dic['logits'], dev_output_dic['targets'])
    test_loss = self.compute_loss(test_output_dic['logits'], test_output_dic['targets'])

    tf.summary.scalar("loss", train_loss, family="train")
    tf.summary.scalar("loss", dev_loss, family="dev")
    tf.summary.scalar("loss", test_loss, family="test")

    update_op, learning_rate = self.get_train_op(train_loss, train_output_dic["trainable_vars"],
                                                 start_learning_rate=0.000,
                                                 base_learning_rate=self.model.hparams.learning_rate,
                                                 warmup_steps=self.model.hparams.learning_rate_warmup_steps,
                                                 clip_gradient_norm=self.model.hparams.clip_grad_norm,
                                                 l2_rate=self.config.l2_rate,
                                                 optimizer="adam"
                                                 )
    tf.summary.scalar("learning_rate", learning_rate, family="train")

    scaffold = tf.train.Scaffold(local_init_op=tf.group(tf.local_variables_initializer(),
                                                        train_iterator.initializer,
                                                        dev_iterator.initializer,
                                                        test_iterator.initializer),
                                 init_feed_dict={})

    return update_op, scaffold, train_output_dic, dev_output_dic, test_output_dic

