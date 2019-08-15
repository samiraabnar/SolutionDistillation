import tensorflow as tf
import numpy as np

from distill.common.hparams import TransformerHparam, LSTMHparam
from distill.data_util.prep_ptb import PTB
from distill.models.lm_lstm import LmLSTM
from distill.layers.lstm import LSTM
from distill.layers.bilstm import BiLSTM
import os

tf.logging.set_verbosity(tf.logging.INFO)


tf.app.flags.DEFINE_string("exp_name", "trial", "")
tf.app.flags.DEFINE_string("task_name", "ptb_lm", "")
tf.app.flags.DEFINE_string("log_dir", "logs", "")
tf.app.flags.DEFINE_string("save_dir", None, "")

tf.app.flags.DEFINE_string("model_type", "plain_lstm", "")
tf.app.flags.DEFINE_integer("hidden_dim", 128, "")
tf.app.flags.DEFINE_integer("output_dim", 8000, "")
tf.app.flags.DEFINE_integer("input_dim", 8000, "")
tf.app.flags.DEFINE_integer("number_of_heads", 4, "")
tf.app.flags.DEFINE_integer("ff_filter_size", 512, "")
tf.app.flags.DEFINE_float("initializer_gain", 1.0, "")
tf.app.flags.DEFINE_float("label_smoothing", 0.1, "")
tf.app.flags.DEFINE_boolean('train_embeddings', True, " False | True")
tf.app.flags.DEFINE_string('sent_rep_mode', "final", "none | final | all")
tf.app.flags.DEFINE_string('attention_mechanism',None, 'attention_mechanism')
tf.app.flags.DEFINE_integer("depth", 1, "")
tf.app.flags.DEFINE_string("sent_rep_mode", 'all', "all| final| ")


tf.app.flags.DEFINE_float("input_dropout_keep_prob", 0.75, "")
tf.app.flags.DEFINE_float("hidden_dropout_keep_prob", 0.5, "")

tf.app.flags.DEFINE_float("learning_rate", 0.001, "")
tf.app.flags.DEFINE_float("l2_rate", 0.001, "")

tf.app.flags.DEFINE_integer("batch_size", 32, "")
tf.app.flags.DEFINE_integer("training_iterations", 30000, "")

tf.app.flags.DEFINE_integer("vocab_size", 8000, "")
tf.app.flags.DEFINE_integer("embedding_dim", 300, "embeddings dim")
tf.app.flags.DEFINE_boolean("bidirectional", False, "If the LSTM layer is bidirectional")


tf.app.flags.DEFINE_string("pretrained_embedding_path", "data/sst/filtered_glove.txt", "pretrained embedding path")
tf.app.flags.DEFINE_string("data_path", "./data", "data path")


hparams = tf.app.flags.FLAGS


class PTBTrainer(object):
  def __init__(self, hparams, model_class):
    self.config = hparams
    self.ptb = PTB("data/ptb")
    self.config.vocab_size = len(self.ptb.word2id)

    if hparams.bidirectional:
      lstm = BiLSTM
    else:
      lstm = LSTM

    self.lm_lstm = model_class(self.config, model=lstm)

  def get_train_op(self, loss, params):

    self.global_step = tf.train.get_or_create_global_step()

    loss_l2 = tf.add_n([tf.nn.l2_loss(p) for p in params]) * self.config.l2_rate

    loss += loss_l2

    base_learning_rate = 0.001
    start_learning_rate = 0.0005
    warmup_steps = 1000
    slope = (base_learning_rate - start_learning_rate) / warmup_steps
    warmup_rate = slope * tf.cast(self.global_step,
                                  tf.float32) + start_learning_rate

    decay_learning_rate = tf.train.exponential_decay(base_learning_rate, self.global_step,
                                                     1000, 0.96, staircase=False)
    learning_rate = tf.where(self.global_step < warmup_steps, warmup_rate,
                             decay_learning_rate)

    #learning_rate = 0.0002
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
    grads_and_vars = opt.compute_gradients(loss, params)
    gradients, variables = zip(*grads_and_vars)
    #self.gradient_norm = tf.global_norm(gradients)
    clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5)
    #self.param_norm = tf.global_norm(params)

    # Include batch norm mean and variance in gradient descent updates
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      # Fetch self.updates to apply gradients to all trainable parameters.
      updates = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)

    return updates, learning_rate

  def get_data_itaratoes(self):
    dataset = tf.data.TFRecordDataset(PTB.get_tfrecord_path("data/ptb", mode="train"))
    dataset = dataset.map(PTB.parse_ptb_examples)
    dataset = dataset.padded_batch(self.config.batch_size, padded_shapes=PTB.get_padded_shapes(), drop_remainder=True)
    dataset = dataset.shuffle(buffer_size=32)
    dataset = dataset.repeat()
    iterator = dataset.make_initializable_iterator()

    dev_dataset = tf.data.TFRecordDataset(PTB.get_tfrecord_path("data/ptb", mode="dev"))
    dev_dataset = dev_dataset.map(PTB.parse_ptb_examples)
    dev_dataset = dev_dataset.shuffle(buffer_size=1101)
    dev_dataset = dev_dataset.repeat()
    dev_dataset = dev_dataset.padded_batch(1000, padded_shapes=PTB.get_padded_shapes(),
                                           drop_remainder=True)
    dev_iterator = dev_dataset.make_initializable_iterator()

    test_dataset = tf.data.TFRecordDataset(PTB.get_tfrecord_path("data/ptb", mode="test"))
    test_dataset = test_dataset.map(PTB.parse_ptb_examples)
    test_dataset = test_dataset.shuffle(buffer_size=2210)
    test_dataset = test_dataset.repeat()
    test_dataset = test_dataset.padded_batch(1000, padded_shapes=PTB.get_padded_shapes(),
                                           drop_remainder=True)
    test_iterator = test_dataset.make_initializable_iterator()

    return iterator, dev_iterator, test_iterator

  def build_train_graph(self):
    self.lm_lstm.build_graph()

    train_iterator, dev_iterator, test_iterator = self.get_data_itaratoes()
    train_output_dic = self.lm_lstm.apply(train_iterator.get_next())
    tf.summary.scalar("loss", train_output_dic["loss"], family="train")
    tf.summary.scalar("accuracy", train_output_dic["accuracy"], family="train")
    tf.summary.scalar("sequence_accuracy", train_output_dic["sequence_accuracy"], family="train")
    tf.summary.scalar("perplexity", train_output_dic["perplexity"], family="train")

    dev_output_dic = self.lm_lstm.apply(dev_iterator.get_next(), is_train=False)
    tf.summary.scalar("loss", dev_output_dic["loss"], family="dev")
    tf.summary.scalar("accuracy", dev_output_dic["accuracy"], family="dev")
    tf.summary.scalar("sequence_accuracy", dev_output_dic["sequence_accuracy"], family="dev")
    tf.summary.scalar("perplexity", dev_output_dic["perplexity"], family="dev")

    test_output_dic = self.lm_lstm.apply(test_iterator.get_next(), is_train=False)
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

  def train(self):
    update_op, scaffold, train_output_dic, dev_output_dic, test_output_dic = self.build_train_graph()

    # self.global_step = tf.train.get_or_create_global_step()
    with tf.train.MonitoredTrainingSession(checkpoint_dir=self.config.save_dir, scaffold=scaffold) as sess:
      for _ in np.arange(self.config.training_iterations):
        _ = sess.run([update_op],
                 feed_dict={})



if __name__ == '__main__':
  Models = {"lstm_lm": LmLSTM}
  tasks = {'ptb_lm': PTB('data/ptb')}

  hparams.vocab_size = tasks[hparams.task_name].vocab_length
  hparams.output_dim = len(tasks[hparams.task_name].target_vocab)

  transformer_params = TransformerHparam(input_dim=hparams.input_dim,
                                         hidden_dim=hparams.hidden_dim,
                                         output_dim=hparams.output_dim,
                                         encoder_depth=hparams.depth,
                                         decoder_depth=0,
                                         number_of_heads=2,
                                         ff_filter_size=512,
                                         initializer_gain=hparams.initializer_gain,
                                         batch_size=hparams.batch_size,
                                         input_dropout_keep_prob=hparams.input_dropout_keep_prob,
                                         hidden_dropout_keep_prob=hparams.hidden_dropout_keep_prob,
                                         vocab_size=hparams.vocab_size,
                                         label_smoothing=hparams.label_smoothing,
                                         encoder_self_attention_dir=hparams.encoder_attention_dir,
                                         decoder_self_attention_dir="top_down",
                                         decoder_cross_attention_dir="top_down",
                                         train_embeddings=hparams.train_embeddings,
                                         learning_rate=hparams.learning_rate
                                         )

  lstm_params = LSTMHparam(input_dim=hparams.input_dim,
                           hidden_dim=hparams.hidden_dim,
                           output_dim=hparams.output_dim,
                           encoder_depth=hparams.depth,
                           decoder_depth=0,
                           number_of_heads=hparams.number_of_heads,
                           ff_filter_size=hparams.ff_filter_size,
                           initializer_gain=hparams.initializer_gain,
                           batch_size=hparams.batch_size,
                           input_dropout_keep_prob=hparams.input_dropout_keep_prob,
                           hidden_dropout_keep_prob=hparams.hidden_dropout_keep_prob,
                           vocab_size=hparams.vocab_size,
                           label_smoothing=hparams.label_smoothing,
                           attention_mechanism=None,
                           sent_rep_mode=hparams.sent_rep_mode,
                           embedding_dim=300,
                           train_embeddings = hparams.train_embeddings,
                           learning_rate=hparams.learning_rate)


  model_params = {"lstm_lm": lstm_params}


  if hparams.save_dir is None:
    hparams.save_dir = os.path.join(hparams.log_dir,hparams.task_name,
                                    '_'.join([hparams.model,
                                              'depth'+str(model_params[hparams.model].depth),
                                              'hidden_dim'+str(model_params[hparams.model].hidden_dim),
                                              'batch_size'+str(model_params[hparams.model].batch_size),
                                              hparams.exp_name]))

    model = Models[hparams.model](model_params[hparams.model],
                                  task=tasks[hparams.task_name],
                                  scope=hparams.model)
    
  trainer = PTBTrainer(hparams, model_class=LmLSTM)
  trainer.train()