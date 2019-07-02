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

class Trec6(object):

  def __init__(self, data_path, build_vocab=False):
    self.data_path = data_path
    self.eos = '<eos>'
    self.pad = '<pad>'
    self.unk = '<unk>'

    if build_vocab:
      self.build_vocab(os.path.join(self.data_path, "TREC.train.all.txt"))
    self.load_vocab()

  @property
  def share_input_output_embeddings(self):
    return False

  @property
  def vocab_length(self):
    return len(self.id2token)

  @property
  def target_vocab(self):
    return [0,1,2,3,4,5,6]

  @property
  def target_length(self):
    return 1

  @property
  def eos_id(self):
    return self.token2id[self.eos]


  def read_tokens(self, filename):
    sentences, labels = self.read_sentences(filename)
    return [token.lower() for sent in sentences for token in sent]

  def build_vocab(self, filename):
    raw_tokens = self.read_tokens(filename)

    counter = collections.Counter(raw_tokens)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    number_of_reserved_tokens = 3
    tokens, _ = list(zip(*count_pairs))
    token_to_id = dict(zip(tokens, np.arange(len(tokens))+number_of_reserved_tokens))

    token_to_id[self.pad] = 0
    token_to_id[self.eos] = 1
    token_to_id[self.unk] = 2


    id_to_token = {}
    for token,id in token_to_id.items():
      id_to_token[id] = token

    print("vocab size: ", len(token_to_id))
    vocab_dict = {'token_to_id': token_to_id, 'id_to_token': id_to_token}

    np.save(os.path.join(self.data_path, "vocab"), vocab_dict)

  def load_vocab(self):
    vocab_path = os.path.join(self.data_path, 'vocab.npy')
    vocab = np.load(vocab_path, allow_pickle=True).item()

    self.token2id = vocab['token_to_id']
    self.id2token = vocab['id_to_token']

  def encode_sent(self, sent):
    encoded_sent = []
    for token in sent + [self.eos]:
      encoded_sent.append(self.token2id.get(token.lower(), self.token2id[self.unk]))

    return encoded_sent

  @staticmethod
  def decode(encoded_sent, id2token):
    sent = []
    for id in encoded_sent:
      sent.append(id2token[id])

    return sent

  def read_sentences(self, filename):
    with tf.gfile.GFile(filename, "rb") as f:
      sentences = f.readlines()
    labels = list(np.zeros(len(sentences)))

    for i in np.arange(len(sentences)):
      sentences[i] = sentences[i].decode("latin-1").replace("\n", "").split()
      labels[i], sentences[i] = sentences[i][0], sentences[i][1:]

    return sentences, labels

  def read_trec6(self, data_path=None, mode='train'):
    """ Loads Trec6 data.
    """

    data_full_path = os.path.join(data_path, "TREC."+mode+".all.txt")

    sentences, labels = self.read_sentences(data_full_path)


    return sentences, labels

  def get_examples(self, sentences, labels):
    examples = []
    for sent, label in zip(sentences, labels):
      sent = self.encode_sent(sent)
      example = {}
      example['token_ids'] = sent
      example['label'] = [int(label)+1] #Just to avoid confusion with padding!
      example['length'] = len(sent)

      examples.append(example)

    return examples

  def get_tf_example(self,example):
    """Convert our own representation of an example's features to Features class for TensorFlow dataset.
    """
    features = tf.train.Features(feature={
      "inputs": tf.train.Feature(int64_list=tf.train.Int64List(value=example['token_ids'])),
      "targets": tf.train.Feature(int64_list=tf.train.Int64List(value=example['label'])),
      "inputs_length": tf.train.Feature(int64_list=tf.train.Int64List(value=[example['length']])),
      "targets_length": tf.train.Feature(int64_list=tf.train.Int64List(value=[1]))
    })
    return features

  def build_tfrecords(self, data, mode):
    tf_examples = []
    for example in self.get_examples(*data):
       tf_examples.append(self.get_tf_example(example))

    with tf.python_io.TFRecordWriter(os.path.join(self.data_path,mode + ".tfr")) as tf_record_writer:
      for example in tqdm(tf_examples):
        tf_record = tf.train.Example(features=example)
        tf_record_writer.write(tf_record.SerializeToString())

  def parse_examples(self, example):
      """Load an example from TF record format."""
      features = {"inputs_length": tf.FixedLenFeature([], tf.int64),
                  "targets_length": tf.FixedLenFeature([], tf.int64),
                  "inputs": tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
                  "targets": tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
                  }
      parsed_example = tf.parse_single_example(example, features=features)

      inputs_length = parsed_example["inputs_length"]
      targets_lengths = parsed_example['targets_length']
      inputs = parsed_example["inputs"]
      targets = parsed_example["targets"]

      return inputs, targets, inputs_length, targets_lengths

  def get_padded_shapes(self,):
    return [None], [None], [], []

  def get_tfrecord_path(self, mode):
    return os.path.join(self.data_path, mode + ".tfr")


class CharTrec6(Trec6):

  def __init__(self, data_path, build_vocab=False):
    self.data_path = data_path
    self.eos = '<eos>'
    self.pad = '<pad>'
    self.unk = '<unk>'

    if build_vocab:
      self.build_vocab(os.path.join(self.data_path, "TREC.train.all.txt"))
    self.load_vocab()


  def read_sentences(self, filename):
    with tf.gfile.GFile(filename, "rb") as f:
      sentences = f.readlines()
    labels = list(np.zeros(len(sentences)))

    for i in np.arange(len(sentences)):
      sentences[i] = sentences[i].decode("latin-1").replace("\n", "").split(" ", 1)
      labels[i], sentences[i] = sentences[i][0], list(sentences[i][1])

    return sentences, labels


if __name__ == '__main__':
  chartrec6 = CharTrec6(data_path="../../data/char_trec6", build_vocab=True)


  train_data = chartrec6.read_trec6('../../data/char_trec6', "train")
  test_data = chartrec6.read_trec6('../../data/char_trec6', "test")

  print("Length of train: ", len(train_data[0]))
  print("Length of test: ", len(test_data[0]))

  chartrec6.build_tfrecords(train_data, "train")
  chartrec6.build_tfrecords(test_data, "test")

  dataset = tf.data.TFRecordDataset(chartrec6.get_tfrecord_path(mode="test"))
  dataset = dataset.map(chartrec6.parse_examples)
  dataset = dataset.padded_batch(1, padded_shapes=chartrec6.get_padded_shapes())
  iterator = dataset.make_initializable_iterator()

  example = iterator.get_next()
  inputs, targets, input_lengths, target_lengths = example
  global_step = tf.train.get_or_create_global_step()
  scaffold = tf.train.Scaffold(local_init_op=tf.group(tf.local_variables_initializer(),
                                                      iterator.initializer))
  with tf.train.MonitoredTrainingSession(checkpoint_dir='logs', scaffold=scaffold) as sess:
    print(sess.run([inputs, targets, input_lengths, target_lengths]))

