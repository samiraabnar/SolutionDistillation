import tensorflow as tf
from distill.pipelines.basic_trainer import Trainer
from distill.common.metrics import padded_cross_entropy_loss, get_eval_metrics


class Seq2SeqTrainer(Trainer):
  def __init__(self, config, model_obj, task):
    super(Seq2SeqTrainer, self).__init__(config, model_obj)
    self.task = task

  def get_train_data_itaratoes(self):
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

  def compute_loss(self,logits, targets):
    xentropy, weights = padded_cross_entropy_loss(
      logits, targets, self.config.label_smoothing, self.config.output_dim)
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
    train_iterator, dev_iterator, test_iterator = self.get_train_data_itaratoes()

    train_examples = train_iterator.get_next()
    dev_examples = dev_iterator.get_next()
    test_examples = test_iterator.get_next()
    pretrained_embeddings = None
    if self.task.pretrained:
      pretrained_embeddings = self.task.get_pretrained_mat("glove_300")
    self.model.create_vars(reuse=False,pretrained_embeddings=pretrained_embeddings)

    train_output_dic = self.model.apply(train_examples, target_length=self.task.target_length, is_train=True)
    dev_output_dic = self.model.apply(dev_examples, target_length=self.task.target_length, is_train=False)
    test_output_dic = self.model.apply(test_examples, target_length=self.task.target_length, is_train=False)

    train_loss = self.compute_loss(train_output_dic['logits'], train_output_dic['targets'])
    dev_loss = self.compute_loss(dev_output_dic['logits'], dev_output_dic['targets'])
    test_loss = self.compute_loss(test_output_dic['logits'], test_output_dic['targets'])

    train_output_dic['loss'] = train_loss
    tf.summary.scalar("loss", train_loss, family="train")
    tf.summary.scalar("loss", dev_loss, family="dev")
    tf.summary.scalar("loss", test_loss, family="test")

    tf.summary.scalar("length", tf.shape(train_output_dic['logits'])[1], family="train")
    tf.summary.scalar("length", tf.shape(dev_output_dic['logits'])[1], family="dev")
    tf.summary.scalar("length", tf.shape(test_output_dic['logits'])[1], family="test")


    if self.task.target_length == 1:
      batch_size = tf.shape(train_output_dic['targets'])[0]
      tf.summary.scalar("classification_accuracy", tf.reduce_mean(
        tf.cast(tf.equal(
          tf.reshape(tf.argmax(train_output_dic['logits'],axis=-1),(batch_size,1)),
          tf.reshape(train_output_dic['targets'], (batch_size,1))), dtype=tf.float32)),
                        family="train")

    self.add_metric_summaries(train_output_dic['logits'], train_output_dic['targets'], "train")
    self.add_metric_summaries(dev_output_dic['logits'], dev_output_dic['targets'], "dev")
    self.add_metric_summaries(test_output_dic['logits'], test_output_dic['targets'], "test")


    tf.summary.scalar("number_of_training_params",
                      tf.reduce_sum([tf.reduce_prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
    update_op, learning_rate = self.get_train_op(train_loss, train_output_dic["trainable_vars"],
                                                 start_learning_rate=0.0005,
                                                 base_learning_rate=self.model.hparams.learning_rate,
                                                 warmup_steps=self.model.hparams.learning_rate_warmup_steps,
                                                 clip_gradient_norm=self.model.hparams.clip_grad_norm
                                                 )
    tf.summary.scalar("learning_rate", learning_rate, family="train")

    scaffold = tf.train.Scaffold(local_init_op=tf.group(tf.local_variables_initializer(),
                                                        train_iterator.initializer,
                                                        dev_iterator.initializer,
                                                        test_iterator.initializer),
                                 init_feed_dict={})

    return update_op, scaffold, train_output_dic, dev_output_dic, test_output_dic
