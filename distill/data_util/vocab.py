"""For encoding and decoding string input to/from ids
"""

import numpy as np
import os

class Vocab(object):
  def __init__(self, path):
    self.word_to_index = {}
    self.index_to_word = {}
    self.unknown = '<unk>'
    self.add_word(self.unknown, count=0)
    self.path = path
  def add_word(self, word, count=1):
    if word not in self.word_to_index:
      index = len(self.word_to_index)
      self.word_to_index[word] = index
      self.index_to_word[index] = word

  def build_vocab(self, words):
    for word in words:
      self.add_word(word)
    print('{} total unique words'.format(len(self.word_to_index)))

  def encode(self, tokens):
    if type(tokens) is str:
      tokens = [tokens]
    ids = []
    for token in tokens:
      if token not in self.word_to_index:
        token = self.unknown
      ids.append(self.word_to_index[token])

    return ids

  def decode(self, ids):
    if type(ids) is int:
      ids = [ids]
    tokens = []
    for id in ids:
      tokens.append(self.index_to_word[id])

    return tokens

  def save(self):
    save_dic = {
      'word_to_index': self.word_to_index,
      'index_to_word': self.index_to_word,
    }
    np.save(self.path, save_dic)

  def load(self):
    loaded_dic = np.load(self.path + ".npy").item()
    self.word_to_index = loaded_dic['word_to_index']
    self.index_to_word = loaded_dic['index_to_word']

  def exists(self):
    return os.path.isfile(self.path + ".npy")

  def __len__(self):
    return len(self.index_to_word)