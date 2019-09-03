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

class OneBLM(SentWiki):

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
    # self.load_vocab()

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
    return [word for sent in sentences for word in sent]

  def read_sentences(self, filename):
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


  def read_raw_data(self, data_path, mode):
    path = os.path.join(data_path, mode+"/*")

    print(path)
    files = glob.glob(path)
    print(files)
    for filename in files:
      train_data = self.read_sentences(filename)
      print("Length of this part of train: ", len(train_data))
      with open(filename+".clean", "w") as f:
        f.writelines('\n'.join(train_data))




if __name__ == '__main__':
  sentwiki = OneBLM(data_path="data/lm1b", build_vocab=True)


  sentwiki.read_raw_data('data/lm1b', "train")
