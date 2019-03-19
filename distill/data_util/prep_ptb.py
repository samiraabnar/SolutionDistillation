from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys
import numpy as np
import tensorflow as tf
from tqdm import tqdm

Py3 = sys.version_info[0] == 3

class PTB(object):

  def __init__(self, data_path):
    self.data_path = data_path
    self.word2id, self.id2word = PTB.load_vocab(self.data_path)

  @staticmethod
  def read_words(filename):
    sentences = PTB.read_sentences(filename)
    return [word for sent in sentences for word in sent]

  @staticmethod
  def build_vocab(filename, unknown_token='<unk>', start_token='<s>'):
    data = PTB.read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    if unknown_token not in word_to_id:
      word_to_id[unknown_token] = len(word_to_id)

    if start_token not in word_to_id:
      word_to_id[start_token] = len(word_to_id)

    id_to_word = {}
    for word,id in word_to_id.items():
      id_to_word[id] = word


    return word_to_id, id_to_word

  @staticmethod
  def load_vocab(data_path):
    vocab_path = os.path.join(data_path, 'vocab.npy')
    vocab = np.load(vocab_path).item()

    return vocab['word_to_id'], vocab['id_to_word']

  @staticmethod
  def encode_sent(sent, word_to_id, unknown_token='<unk>'):
    encoded_sent = []
    for word in sent:
      if word not in word_to_id:
        word = unknown_token
      encoded_sent.append(word_to_id[word])

    return encoded_sent

  @staticmethod
  def decode(encoded_sent, id_to_word):
    sent = []
    for id in encoded_sent:
      sent.append(id_to_word[id])

    return sent

  @staticmethod
  def read_sentences(filename):
    with tf.gfile.GFile(filename, "r") as f:
      sentences = f.read().replace("\n", "<eos>\n").split("\n")

    for i in np.arange(len(sentences)):
      sentences[i] = ['<s>'] + sentences[i].split()

    return sentences

  @staticmethod
  def read_ptb_raw_data(data_path=None):
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

    train_data = PTB.read_sentences(train_path)
    valid_data = PTB.read_sentences(valid_path)
    test_data = PTB.read_sentences(test_path)

    return train_data, valid_data, test_data

  @staticmethod
  def get_examples(sentences, word_to_id, start_token="<s>"):
    examples = []
    for sent in sentences:
      sent = PTB.encode_sent(sent, word_to_id)
      example = {}
      example['token_ids'] = sent[:len(sent)-1]
      example['labels'] = sent[1:]
      example['lengths'] = len(sent) - 1

      examples.append(example)

    return examples

  @staticmethod
  def get_tf_example(example):
    """Convert our own representation of an example's features to Features class for TensorFlow dataset.
    """
    features = tf.train.Features(feature={
      "token_ids": tf.train.Feature(int64_list=tf.train.Int64List(value=example['token_ids'])),
      "labels": tf.train.Feature(int64_list=tf.train.Int64List(value=example['labels'])),
      "lengths": tf.train.Feature(int64_list=tf.train.Int64List(value=[example['lengths']]))
    })
    return features

  @staticmethod
  def build_tfrecords(data, word_to_id, data_path,  mode):
    tf_examples = []
    for example in PTB.get_examples(data, word_to_id):
       tf_examples.append(PTB.get_tf_example(example))

    with tf.python_io.TFRecordWriter(os.path.join(data_path,mode + ".tfr")) as tf_record_writer:
      for example in tqdm(tf_examples):
        tf_record = tf.train.Example(features=example)
        tf_record_writer.write(tf_record.SerializeToString())


  @staticmethod
  def parse_ptb_examples(example):
      """Load an example from TF record format."""
      features = {"lengths": tf.FixedLenFeature([], tf.int64),
                  "token_ids": tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
                  "labels": tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
                  }
      parsed_example = tf.parse_single_example(example, features=features)

      lengths = parsed_example["lengths"]
      inputs = parsed_example["token_ids"]
      labels = parsed_example["labels"]

      return inputs, labels, lengths


  @staticmethod
  def get_padded_shapes():
    return [None], [None], []

  @staticmethod
  def get_tfrecord_path(datapath, mode):
    return os.path.join(datapath, mode + ".tfr")

if __name__ == '__main__':
  word_to_id, id_to_word = PTB.build_vocab(filename="data/ptb/ptb.train.txt")
  print("vocab size: ", len(word_to_id))
  vocab_dict = {'word_to_id':word_to_id, 'id_to_word':id_to_word }
  np.save("data/ptb/vocab", vocab_dict)

  train_data, dev_data, test_data = PTB.read_ptb_raw_data('data/ptb')
  #word_to_id, id_to_word = load_vocab('data/ptb')

  #print("Length of train: ", len(train_data))
  #print("Length of test: ", len(test_data))
  #print("Length of dev: ", len(dev_data))

  PTB.build_tfrecords(train_data, word_to_id, 'data/ptb', "train")
  PTB.build_tfrecords(test_data, word_to_id, 'data/ptb',"test")
  PTB.build_tfrecords(dev_data, word_to_id, 'data/ptb', "dev")

  dataset = tf.data.TFRecordDataset(PTB.get_tfrecord_path("data/ptb", mode="test"))
  dataset = dataset.map(PTB.parse_ptb_examples)
  dataset = dataset.padded_batch(10, padded_shapes=PTB.get_padded_shapes())
  iterator = dataset.make_initializable_iterator()

  example = iterator.get_next()
  inputs, labels, lengths = example
  global_step = tf.train.get_or_create_global_step()
  scaffold = tf.train.Scaffold(local_init_op=tf.group(tf.local_variables_initializer(),
                                                      iterator.initializer))
  with tf.train.MonitoredTrainingSession(checkpoint_dir='logs', scaffold=scaffold) as sess:
    print(sess.run(inputs))

