from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import glob
import string
import re
import num2words

from distill.data_util.prep_sentwiki import SentWiki

Py3 = sys.version_info[0] == 3

class Lm1b(SentWiki):

  def __init__(self, data_path, build_vocab=False, tie_embeddings=True):
    self.data_path = data_path
    self.eos = '<eos>'
    self.pad = '<pad>'
    self.unk = '<unk>'
    self.start_token = '<s>'
    self.pre_defs = [self.pad, self.start_token, self.eos, self.unk]
    self.tie_embeddings = tie_embeddings

    # if build_vocab:
    #   self.build_vocab(os.path.join(self.data_path, "train.txt"))
    self.load_vocab()

  @property
  def max_length(self):
    return 100

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
    return [word.lower() for sent in sentences for word in sent]

  def read_and_clean_sentences(self, filename):
    sentences = []
    with tf.gfile.GFile(filename, "r") as f:
      sentences.extend(f.read().split("\n"))

    transtable = str.maketrans('', '', string.punctuation)
    translates_sentences = []

    for i in np.arange(len(sentences)):
      sentence = []
      for word in sentences[i].translate(transtable).split():
        if re.match("^\d+?\.*\d+?$", word) is None and word.isdigit:
          sentence.append(word)
        else:
          sentence.append(num2words.num2words(word))

      translates_sentences.append(' '.join(sentence))

    return translates_sentences


  def clean_data(self, data_path, mode):
    path = os.path.join(data_path, mode+"/*")

    print(path)
    files = glob.glob(path)
    print(files)
    for filename in files:
      train_data = self.read_and_clean_sentences(filename)
      print("Length of this part of train: ", len(train_data))
      with open(filename+".clean", "w") as f:
        f.writelines('\n'.join(train_data))

  def build_vocab(self, data_path):
    path = os.path.join(data_path + "/*.clean")

    print(path)
    files = glob.glob(path)
    print(files)
    counter = collections.Counter()
    for filename in files:
      data = self.read_words(filename)
      counter.update(data)


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


  def build_tfrecords(self, data, mode, file_name):
    tf_examples = []
    for example in self.get_examples(data):
       tf_examples.append(self.get_tf_example(example))

    with tf.python_io.TFRecordWriter(os.path.join(self.data_path,file_name + ".tfr")) as tf_record_writer:
      for example in tqdm(tf_examples):
        tf_record = tf.train.Example(features=example)
        tf_record_writer.write(tf_record.SerializeToString())

  def build_all_tfrecords(self, path, mode):
    path = os.path.join(path, mode,"*.clean")
    print(path)
    files = glob.glob(path)
    print(files)
    for filename in files:
      data = self.read_sentences(filename)
      self.build_tfrecords(data,mode, '/'.join(filename.split("/")[-2:]))

  def get_tfrecord_path(self, mode):
    path = os.path.join(self.data_path, mode, "*.tfr")
    return glob.glob(path)

if __name__ == '__main__':
  lm1b = Lm1b(data_path="data/lm1b", build_vocab=True)
  lm1b.clean_data('data/lm1b', "test")
  #lm1b.build_vocab('data/lm1b/train')
  lm1b.build_all_tfrecords('data/lm1b', "test")

  print(lm1b.get_tfrecord_path(mode="test"))
  dataset = tf.data.TFRecordDataset(lm1b.get_tfrecord_path(mode="test"))
  dataset = dataset.map(lm1b.parse_examples)
  dataset = dataset.padded_batch(1, padded_shapes=lm1b.get_padded_shapes())
  iterator = dataset.make_initializable_iterator()

  example = iterator.get_next()
  inputs, targets, input_lengths, target_lengths = example
  global_step = tf.train.get_or_create_global_step()
  scaffold = tf.train.Scaffold(local_init_op=tf.group(tf.local_variables_initializer(),
                                                      iterator.initializer))
  with tf.train.MonitoredTrainingSession(checkpoint_dir='logs', scaffold=scaffold) as sess:
    print(sess.run([inputs, targets, input_lengths, target_lengths]))
