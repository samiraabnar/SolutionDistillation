import tensorflow as tf

from distill.common.metrics import padded_cross_entropy_loss, get_eval_metrics
from distill.pipelines.basic_trainer import Trainer



class LMTrainer(Trainer):
  def __init__(self, config, model_obj, task):
    super(LMTrainer, self).__init__(config, model_obj)
    self.task = task

  def get_data_itarators(self):
    dataset = tf.data.TFRecordDataset(self.task.get_tfrecord_path(mode="train"))
    dataset = dataset.map(self.task.parse_examples)
    dataset = dataset.padded_batch(self.config.batch_size, padded_shapes=self.task.get_padded_shapes())
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.repeat()
    train_iterator = dataset.make_initializable_iterator()

    dataset = tf.data.TFRecordDataset(self.task.get_tfrecord_path(mode="dev"))
    dataset = dataset.map(self.task.parse_examples)
    dataset = dataset.padded_batch(self.config.batch_size, padded_shapes=self.task.get_padded_shapes())
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.repeat()
    dev_iterator = dataset.make_initializable_iterator()

    dataset = tf.data.TFRecordDataset(self.task.get_tfrecord_path(mode="test"))
    dataset = dataset.map(self.task.parse_examples)
    dataset = dataset.padded_batch(self.config.batch_size, padded_shapes=self.task.get_padded_shapes())
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

    tf.summary.scalar("loss", train_output_dic["loss"], family="train")
    tf.summary.scalar("accuracy", train_output_dic["accuracy"], family="train")
    tf.summary.scalar("sequence_accuracy", train_output_dic["sequence_accuracy"], family="train")
    tf.summary.scalar("perplexity", train_output_dic["perplexity"], family="train")

    self.add_metric_summaries(train_output_dic['logits'], train_output_dic['targets'], "train")

    dev_output_dic = self.model.apply(dev_iterator.get_next(), is_train=False)
    tf.summary.scalar("loss", dev_output_dic["loss"], family="dev")
    tf.summary.scalar("accuracy", dev_output_dic["accuracy"], family="dev")
    tf.summary.scalar("sequence_accuracy", dev_output_dic["sequence_accuracy"], family="dev")
    tf.summary.scalar("perplexity", dev_output_dic["perplexity"], family="dev")

    test_output_dic = self.model.apply(test_iterator.get_next(), is_train=False)
    tf.summary.scalar("loss", test_output_dic["loss"], family="test")
    tf.summary.scalar("accuracy", test_output_dic["accuracy"], family="test")
    tf.summary.scalar("sequence_accuracy", test_output_dic["sequence_accuracy"], family="test")
    tf.summary.scalar("perplexity", test_output_dic["perplexity"], family="test")

    train_loss = train_output_dic["loss"]
    update_op, learning_rate = self.get_train_op(train_loss, train_output_dic["trainable_vars"],
                                                 start_learning_rate=0.000,
                                                 base_learning_rate=self.model.hparams.learning_rate,
                                                 warmup_steps=self.model.hparams.learning_rate_warmup_steps,
                                                 clip_gradient_norm=self.model.hparams.clip_grad_norm,
                                                 l2_rate=self.config.l2_rate
                                                 )
    tf.summary.scalar("learning_rate", learning_rate, family="train")

    scaffold = tf.train.Scaffold(local_init_op=tf.group(tf.local_variables_initializer(),
                                                        train_iterator.initializer,
                                                        dev_iterator.initializer,
                                                        test_iterator.initializer),
                                 init_feed_dict={})

    return update_op, scaffold, train_output_dic, dev_output_dic, test_output_dic

