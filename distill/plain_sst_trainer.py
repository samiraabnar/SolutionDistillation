import tensorflow as tf
import numpy as np
from distill.data_util.prep_sst import SST
from distill.models.sentiment_lstm import SentimentLSTM
from distill.models.sentiment_tree_lstm import SentimentTreeLSTM

from distill.common.util import cosine_decay_with_warmup
from distill.data_util.vocab import PretrainedVocab
import os

tf.logging.set_verbosity(tf.logging.INFO)


tf.app.flags.DEFINE_string("exp_name", "trial", "")
tf.app.flags.DEFINE_string("task_name", "sst", "")
tf.app.flags.DEFINE_string("log_dir", "logs", "")
tf.app.flags.DEFINE_string("save_dir", None, "")

tf.app.flags.DEFINE_string("model_type", "plain_lstm", "")
tf.app.flags.DEFINE_integer("hidden_dim", 50, "")
tf.app.flags.DEFINE_integer("depth", 1, "")
tf.app.flags.DEFINE_integer("input_dim", None, "")
tf.app.flags.DEFINE_integer("output_dim", 2, "")

tf.app.flags.DEFINE_string("loss_type", "root_loss", "")
tf.app.flags.DEFINE_float("input_dropout_keep_prob", 0.75, "")
tf.app.flags.DEFINE_float("hidden_dropout_keep_prob", 0.5, "")

tf.app.flags.DEFINE_float("learning_rate", 0.05, "")
tf.app.flags.DEFINE_float("l2_rate", 0.001, "")

tf.app.flags.DEFINE_integer("batch_size", 32, "")
tf.app.flags.DEFINE_integer("training_iterations", 12000, "")

tf.app.flags.DEFINE_integer("vocab_size", 8000, "")
tf.app.flags.DEFINE_integer("embedding_dim", 300, "embeddings dim")


tf.app.flags.DEFINE_string("pretrained_embedding_path", "/Users/samiraabnar/Codes/Data/word_embeddings/glove.6B/glove.6B.300d.txt", "pretrained embedding path")
tf.app.flags.DEFINE_string("data_path", "./data", "data path")


hparams = tf.app.flags.FLAGS




class PlainSSTTrainer(object):
  def __init__(self, hparams, model_class):
    self.config = hparams
    self.sst = SST("data/sst", pretrained_path=self.config.pretrained_embedding_path, embedding_size=self.config.embedding_dim)
    self.config.input_dim = len(self.sst.vocab)

    self.vocab = PretrainedVocab(self.config.data_path, self.config.pretrained_embedding_path,
                                 self.config.embedding_dim)
    self.pretrained_word_embeddings, self.word2id = self.vocab.get_word_embeddings()
    self.sentimen_lstm = model_class(self.config)

  def get_train_op(self, loss, params):
    # add training op
    # Learning rate is linear from step 0 to self.FLAGS.lr_warmup. Then it decays as 1/sqrt(timestep).

    self.global_step = tf.train.get_or_create_global_step()

    opt = tf.train.AdadeltaOptimizer(learning_rate=0.05)
    grads_and_vars = opt.compute_gradients(loss, params)
    gradients, variables = zip(*grads_and_vars)
    self.gradient_norm = tf.global_norm(gradients)
    clipped_gradients, _ = tf.clip_by_global_norm(gradients, 10)
    self.param_norm = tf.global_norm(params)

    # Include batch norm mean and variance in gradient descent updates
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      # Fetch self.updates to apply gradients to all trainable parameters.
      updates = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)

    return updates, self.config.learning_rate

  def get_data_itaratoes(self):
    dataset = tf.data.TFRecordDataset(SST.get_tfrecord_path("data/sst", mode="train"))
    dataset = dataset.map(SST.parse_full_sst_tree_examples)
    dataset = dataset.padded_batch(self.config.batch_size, padded_shapes=SST.get_padded_shapes(), drop_remainder=True)
    dataset = dataset.shuffle(buffer_size=8544)
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
    self.pretrained_embeddings_ph = tf.placeholder(tf.float32, shape=(self.config.input_dim, self.config.embedding_dim))
    self.sentimen_lstm.build_graph(self.pretrained_embeddings_ph)

    train_iterator, dev_iterator, test_iterator = self.get_data_itaratoes()
    train_output_dic = self.sentimen_lstm.apply(train_iterator.get_next())
    tf.summary.scalar("loss", train_output_dic[self.config.loss_type], family="train")
    tf.summary.scalar("accuracy", train_output_dic["root_accuracy"], family="train")
    tf.summary.scalar("total_matchings", train_output_dic["total_matchings"], family="train")
    tf.summary.scalar("positive_ratio", tf.reduce_mean(tf.cast(train_output_dic['labels'], tf.float32)), family="train")
    tf.summary.scalar("predicted_positive_ratio", tf.reduce_mean(tf.cast(train_output_dic['predictions'], tf.float32)), family="train")


    dev_output_dic = self.sentimen_lstm.apply(dev_iterator.get_next())
    tf.summary.scalar("loss", dev_output_dic[self.config.loss_type], family="dev")
    tf.summary.scalar("accuracy", dev_output_dic["root_accuracy"], family="dev")
    tf.summary.scalar("total_matchings", dev_output_dic["total_matchings"], family="dev")
    tf.summary.scalar("predicted_positive_ratio", tf.reduce_mean(tf.cast(dev_output_dic['predictions'], tf.float32)), family="dev")

    test_output_dic = self.sentimen_lstm.apply(test_iterator.get_next())
    tf.summary.scalar("loss", test_output_dic[self.config.loss_type], family="test")
    tf.summary.scalar("accuracy", test_output_dic["root_accuracy"], family="test")
    tf.summary.scalar("total_matchings", test_output_dic["total_matchings"], family="test")
    tf.summary.scalar("predicted_positive_ratio", tf.reduce_mean(tf.cast(test_output_dic['predictions'], tf.float32)),
                      family="test")

    update_op, learning_rate = self.get_train_op(train_output_dic[self.config.loss_type],train_output_dic["trainable_vars"])
    tf.summary.scalar("learning_rate", learning_rate, family="train")

    scaffold = tf.train.Scaffold(local_init_op=tf.group(tf.local_variables_initializer(),
                                                        train_iterator.initializer,
                                                        dev_iterator.initializer,
                                                        test_iterator.initializer),
                                 init_feed_dict={self.pretrained_embeddings_ph: self.pretrained_word_embeddings})

    return update_op, scaffold, train_output_dic

  def train(self):
    update_op, scaffold, train_output_dic = self.build_train_graph()

    # self.global_step = tf.train.get_or_create_global_step()
    with tf.train.MonitoredTrainingSession(checkpoint_dir=self.config.save_dir, scaffold=scaffold) as sess:
      for _ in np.arange(self.config.training_iterations):
        sess.run(update_op,
                 feed_dict={self.pretrained_embeddings_ph: self.pretrained_word_embeddings})



if __name__ == '__main__':
  if hparams.save_dir is None:
    hparams.save_dir = os.path.join(hparams.log_dir,hparams.task_name, '_'.join([hparams.model_type, 'depth'+str(hparams.depth),'hidden_dim'+str(hparams.hidden_dim),hparams.exp_name]))
  trainer = PlainSSTTrainer(hparams, model_class=SentimentLSTM)
  trainer.train()