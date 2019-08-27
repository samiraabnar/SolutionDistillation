"""For encoding and decoding string input to/from ids
"""

import numpy as np
import os
from tqdm import tqdm

def get_word_embs(word_emb_path, word_emb_size, predefineds, vocabulary_size=1000000):
  """Reads from preprocessed GloVe .txt file and returns embedding matrix and
  mappings from words to word ids.
  Input:
    word_emb_path: string. Path to preprocessed glove file.
    vec_size: int. Dimensionality of a word vector.
  Returns:
    word_emb_matrix: Numpy array shape (vocab_size, vec_size) containing word embeddings.
      Only includes embeddings for words that were seen in the dev/train sets.
    word2id: dictionary mapping word (string) to word id (int)
  """

  print("Loading word embeddings from file: {}...".format(word_emb_path))

  word_emb_matrix = []
  word2id = {}
  idx = 0
  with open(word_emb_path, 'r', encoding='utf-8') as fh:
    for line in tqdm(fh, total=vocabulary_size):
      line = line.rstrip().split(" ")
      word = line[0]
      vector = list(map(float, line[1:]))
      if word_emb_size != len(vector):
        raise Exception(word+": Expected vector of size {}, but got vector of size {}.".format(word_emb_size, len(vector)))
      word_emb_matrix.append(vector)
      word2id[word] = idx
      idx += 1
    for word in predefineds:
      word2id[word] = idx
      word_emb_matrix.append(predefineds[word])
      idx += 1

  word_emb_matrix = np.array(word_emb_matrix, dtype=np.float32)
  print("Loaded word embedding matrix with shape {}.".format(word_emb_matrix.shape))

  return word_emb_matrix, word2id


class Vocab(object):
  def __init__(self, path, init_tokens=['<pad>', '<eos>','<unk>','<cls>']):
    self.word_to_index = {}
    self.index_to_word = {}

    self.build_vocab(init_tokens)
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

  def encode(self, tokens, unknown='<unk>'):
    if type(tokens) is str:
      tokens = [tokens]
    ids = []
    for token in tokens:
      if token not in self.word_to_index:
        token = unknown
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
    print("loading vocab from: ",self.path)
    loaded_dic = np.load(self.path + ".npy", allow_pickle=True).item()
    self.word_to_index = loaded_dic['word_to_index']
    self.index_to_word = loaded_dic['index_to_word']



  def exists(self):
    return os.path.isfile(self.path + ".npy")

  def __len__(self):
    return len(self.index_to_word)


class PretrainedVocab(Vocab):
  def __init__(self, path, pre_training_path, embedding_dim,
               init_tokens={'pad':'<pad>', 'eos':'<eos>', 'unk':'<unk>', 'cls': '<cls>'}):
    self.word_to_index = {}
    self.index_to_word = {}
    self.unknown = init_tokens['unk']
    self.pad = init_tokens['pad']
    self.eos = init_tokens['eos']
    self.cls = init_tokens['cls']

    self.predefineds = {self.unknown: np.random.uniform(
      -0.05, 0.05, embedding_dim).astype(np.float32),
                        self.pad:np.random.uniform(
      -0.05, 0.05, embedding_dim).astype(np.float32),
                        self.eos:np.random.uniform(
      -0.05, 0.05, embedding_dim).astype(np.float32),
                        self.cls: np.random.uniform(
                          -0.05, 0.05, embedding_dim).astype(np.float32)
                        }

    self.path = path
    self.pre_training_path = pre_training_path
    self.dimension = embedding_dim

  def build_vocab(self, words):
    word_embedding_mat, self.word_to_index = get_word_embs(self.pre_training_path, self.dimension, self.predefineds)
    self.index_to_word = {}
    for word in self.word_to_index:
      self.index_to_word[self.word_to_index[word]] = word

    self.add_word(self.unknown, count=0)
    print('{} total unique words'.format(len(self.word_to_index)))

  def get_word_embeddings(self):
   return get_word_embs(self.pre_training_path, self.dimension, self.predefineds)

  def save(self):
    save_dic = {
      'word_to_index': self.word_to_index,
      'index_to_word': self.index_to_word,
      'predefineds': self.predefineds
    }
    np.save(self.path, save_dic)

  def load(self):
    loaded_dic = np.load(self.path + ".npy", allow_pickle=True).item()
    self.word_to_index = loaded_dic['word_to_index']
    self.index_to_word = loaded_dic['index_to_word']
    self.predefineds = loaded_dic['predefineds']

  def exists(self):
    return os.path.isfile(self.path + ".npy")

  def __len__(self):
    return len(self.index_to_word)