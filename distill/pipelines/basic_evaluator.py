import tensorflow as tf
import numpy as np

class Evaluator(object):
  def __init__(self, config, model_obj):
    self.config = config
    self.model = model_obj

  def eval(self):
    dev_scaffold, dev_iterator = self.get_dev_itarator()
    test_scaffold, test_iterator = self.get_test_itarator()
    dev_output_dic = self.build_eval_graph(dev_iterator)
    test_output_dic = self.build_eval_graph(test_iterator)

    with tf.train.MonitoredTrainingSession(checkpoint_dir=self.config.save_dir, scaffold=dev_scaffold) as sess:
      accuracy_list = []
      while not sess.should_stop():
        accuracy_list.append(sess.run(dev_output_dic['accuracy']))

      print("dev accuracy:", np.mean(accuracy_list))

    with tf.train.MonitoredTrainingSession(checkpoint_dir=self.config.save_dir, scaffold=test_scaffold) as sess:
      accuracy_list = []
      while not sess.should_stop():
        accuracy_list.append(sess.run(test_output_dic['accuracy']))

      print("test accuracy:", np.mean(accuracy_list))




  def get_dev_itarator(self):
    raise NotImplementedError()

  def get_test_itarator(self):
    raise NotImplementedError()

  def build_eval_graph(self, iterator):
    raise NotImplementedError()