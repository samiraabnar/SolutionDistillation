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

class SentWiki(object):

  def __init__(self, data_path, build_vocab=False, tie_embeddings=True):
    self.data_path = data_path
    self.eos = '<eos>'
    self.pad = '<pad>'
    self.unk = '<unk>'
    self.start_token = '<s>'
    self.pre_defs = [self.pad, self.start_token, self.eos, self.unk]
    self.tie_embeddings = tie_embeddings

    if build_vocab:
      self.build_vocab(os.path.join(self.data_path, "train.txt"))
    self.load_vocab()

  @property
  def if_label_gaussian_noise(self):
    return False

  @property
  def guassian_noise_scale(self):
    return None

  @property
  def share_input_output_embeddings(self):
    return self.tie_embeddings

  @property
  def vocab_length(self):
    return len(self.id2word)

  @property
  def target_vocab(self):
    return self.id2word

  @property
  def target_length(self):
    return None

  @property
  def eos_id(self):
    return self.word2id[self.eos]


  def read_words(self, filename):
    sentences = self.read_sentences(filename)
    return [word for sent in sentences for word in sent]

  def build_vocab(self, filename):
    data = self.read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    words = self.pre_defs + list(words)
    print(words[0])
    print(words[1])
    print(words[2])
    print(words[3])
    print(words[4])
    word_to_id = dict(zip(words, range(len(words))))

    id_to_word = {}
    for word,id in word_to_id.items():
      id_to_word[id] = word

    print("vocab size: ", len(word_to_id))
    vocab_dict = {'word_to_id': word_to_id, 'id_to_word': id_to_word}

    np.save(os.path.join(self.data_path, "vocab"), vocab_dict)

  def load_vocab(self):
    vocab_path = os.path.join(self.data_path, 'vocab.npy')
    vocab = np.load(vocab_path, allow_pickle=True).item()

    self.word2id = vocab['word_to_id']
    self.id2word = vocab['id_to_word']

  def encode_sent(self, sent):
    encoded_sent = [self.word2id[self.start_token]]
    for word in sent:
      if word not in self.word2id:
        word = self.unk
      encoded_sent.append(self.word2id[word])

    encoded_sent.append(self.word2id[self.eos])
    return encoded_sent

  @staticmethod
  def decode(encoded_sent, id_to_word):
    sent = []
    for id in encoded_sent:
      sent.append(id_to_word[id])

    return sent

  def read_sentences(self, filename):
    with tf.gfile.GFile(filename, "r") as f:
      sentences = f.read().replace().split("\n")

    for i in np.arange(len(sentences)):
      sentences[i] = sentences[i].split()

    return sentences

  def read_raw_data(self, data_path):

    train_path = os.path.join(data_path, "train.txt")
    valid_path = os.path.join(data_path, "val.txt")
    test_path = os.path.join(data_path, "test.txt")

    train_data = self.read_sentences(train_path)
    valid_data = self.read_sentences(valid_path)
    test_data = self.read_sentences(test_path)

    return train_data, valid_data, test_data

  def get_examples(self, sentences):
    examples = []
    for sent in sentences:
      sent = self.encode_sent(sent)
      example = {}
      example['token_ids'] = sent[:len(sent)-1]
      example['labels'] = sent[1:]
      example['lengths'] = len(sent) - 1

      examples.append(example)

    return examples

  def get_tf_example(self,example):
    """Convert our own representation of an example's features to Features class for TensorFlow dataset.
    """
    features = tf.train.Features(feature={
      "inputs": tf.train.Feature(int64_list=tf.train.Int64List(value=example['token_ids'])),
      "targets": tf.train.Feature(int64_list=tf.train.Int64List(value=example['labels'])),
      "inputs_length": tf.train.Feature(int64_list=tf.train.Int64List(value=[example['lengths']])),
      "targets_length": tf.train.Feature(int64_list=tf.train.Int64List(value=[example['lengths']]))
    })
    return features

  def build_tfrecords(self, data, mode):
    tf_examples = []
    for example in self.get_examples(data):
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


if __name__ == '__main__':
  sentwiki = SentWiki(data_path="data/sent_wiki", build_vocab=True)


  train_data, dev_data, test_data = sentwiki.read_raw_data('data/sent_wiki')

  print("Length of train: ", len(train_data))
  print("Length of test: ", len(test_data))
  print("Length of dev: ", len(dev_data))

  sentwiki.build_tfrecords(train_data, "train")
  sentwiki.build_tfrecords(test_data, "test")
  sentwiki.build_tfrecords(dev_data, "dev")

  dataset = tf.data.TFRecordDataset(sentwiki.get_tfrecord_path(mode="test"))
  dataset = dataset.map(sentwiki.parse_examples)
  dataset = dataset.padded_batch(1, padded_shapes=sentwiki.get_padded_shapes())
  iterator = dataset.make_initializable_iterator()

  example = iterator.get_next()
  inputs, targets, input_lengths, target_lengths = example
  global_step = tf.train.get_or_create_global_step()
  scaffold = tf.train.Scaffold(local_init_op=tf.group(tf.local_variables_initializer(),
                                                      iterator.initializer))
  with tf.train.MonitoredTrainingSession(checkpoint_dir='logs', scaffold=scaffold) as sess:
    print(sess.run([inputs, targets, input_lengths, target_lengths]))

