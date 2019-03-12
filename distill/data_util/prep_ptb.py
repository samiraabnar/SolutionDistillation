from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys

import tensorflow as tf

Py3 = sys.version_info[0] == 3

def read_words(filename):
  with tf.gfile.GFile(filename, "r") as f:
    if Py3:
      return f.read().replace("\n", "<eos>").split()
    else:
      return f.read().decode("utf-8").replace("\n", "<eos>").split()


def build_vocab(filename):
  data = read_words(filename)

  counter = collections.Counter(data)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

  words, _ = list(zip(*count_pairs))
  word_to_id = dict(zip(words, range(len(words))))

  return word_to_id

def encode_sent(sent, word_to_id, unknown_token='<unk>'):
  encoded_sent = []
  for word in sent:
    if word not in word_to_id:
      word = unknown_token
    encoded_sent.append(word_to_id[word])

  return encoded_sent

def read_examples(filename, word_to_id):
  with tf.gfile.GFile(filename, "r") as f:
    sentences = f.read().replace("\n", "<eos>\n").split("\n")
    encoded_sentences = map(lambda sent: encode_sent(sent, word_to_id), sentences)

  return encoded_sentences


def ptb_raw_data(data_path=None):
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

  word_to_id = build_vocab(train_path)
  train_data = read_examples(train_path, word_to_id)
  valid_data = read_examples(valid_path, word_to_id)
  test_data = read_examples(test_path, word_to_id)
  vocabulary = len(word_to_id)

  return train_data, valid_data, test_data, vocabulary