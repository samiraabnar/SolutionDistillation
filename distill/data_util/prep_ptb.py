from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from distill.data_util.prep_sentwiki import SentWiki

Py3 = sys.version_info[0] == 3

class PTB(SentWiki):
  def __init__(self, data_path, build_vocab=False, tie_embeddings=True):

    self.data_path = data_path
    self.eos = '<eos>'
    self.pad = '<pad>'
    self.unk = '<unk>'
    self.start_token = '<s>'
    self.pre_defs = [self.pad, self.start_token, self.eos, self.unk]
    self.tie_embeddings = tie_embeddings

    if build_vocab:
      self.build_vocab(os.path.join(self.data_path, "ptb.train.txt"))
    self.load_vocab()


  def read_raw_data(self, data_path=None):
    """Load PTB raw data from data directory "data_path".
    Reads PTB text files, converts strings to integer ids,
    and performs mini-batching of the inputs.
    The PTB dataset comes from Tomas Mikolov's webpage:
    http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
    Args:
      data_path: string path to the directory where simple-examples.tgz has
        been extracted.
    Returns:
      tuple (train_data, valid_data, test_data, vocabulary)
      where each of the data objects can be passed to PTBIterator.
    """

    train_path = os.path.join(data_path, "ptb.train.txt")
    valid_path = os.path.join(data_path, "ptb.valid.txt")
    test_path = os.path.join(data_path, "ptb.test.txt")

    train_data = self.read_sentences(train_path)
    valid_data = self.read_sentences(valid_path)
    test_data = self.read_sentences(test_path)

    return train_data, valid_data, test_data




if __name__ == '__main__':
  ptb = PTB(data_path="data/ptb", build_vocab=True)


  train_data, dev_data, test_data = ptb.read_raw_data('data/ptb')
  #word_to_id, id_to_word = load_vocab('data/ptb')

  print("Length of train: ", len(train_data))
  print("Length of test: ", len(test_data))
  print("Length of dev: ", len(dev_data))

  ptb.build_tfrecords(train_data, "train")
  ptb.build_tfrecords(test_data, "test")
  ptb.build_tfrecords(dev_data, "dev")

  dataset = tf.data.TFRecordDataset(ptb.get_tfrecord_path(mode="test"))
  dataset = dataset.map(ptb.parse_examples)
  dataset = dataset.padded_batch(1, padded_shapes=ptb.get_padded_shapes())
  iterator = dataset.make_initializable_iterator()

  example = iterator.get_next()
  inputs, targets, input_lengths, target_lengths = example
  global_step = tf.train.get_or_create_global_step()
  scaffold = tf.train.Scaffold(local_init_op=tf.group(tf.local_variables_initializer(),
                                                      iterator.initializer))
  with tf.train.MonitoredTrainingSession(checkpoint_dir='logs', scaffold=scaffold) as sess:
    print(sess.run([inputs, targets, input_lengths, target_lengths]))

