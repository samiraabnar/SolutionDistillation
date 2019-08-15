import tensorflow as tf
from distill.pipelines.basic_trainer import Trainer



class LMTrainer(Trainer):
  def __init__(self, config, model_obj, task):
    super(LMTrainer, self).__init__(config, model_obj)
    self.task = task

  def build_train_graph(self):
    self.lm_lstm.build_graph()

    train_iterator, dev_iterator, test_iterator = self.get_data_itaratoes()
    train_output_dic = self.model.apply(train_iterator.get_next())
    tf.summary.scalar("loss", train_output_dic["loss"], family="train")
    tf.summary.scalar("accuracy", train_output_dic["accuracy"], family="train")
    tf.summary.scalar("sequence_accuracy", train_output_dic["sequence_accuracy"], family="train")
    tf.summary.scalar("perplexity", train_output_dic["perplexity"], family="train")

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

    update_op, learning_rate = self.get_train_op(train_output_dic["loss"],train_output_dic["trainable_vars"])
    tf.summary.scalar("learning_rate", learning_rate, family="train")

    scaffold = tf.train.Scaffold(local_init_op=tf.group(tf.local_variables_initializer(),
                                                        train_iterator.initializer,
                                                        dev_iterator.initializer,
                                                        test_iterator.initializer),
                                 init_feed_dict={})

    return update_op, scaffold, train_output_dic, dev_output_dic, test_output_dic
