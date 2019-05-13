from tensorflow import keras
import tensorflow as tf
import os
import numpy as np
from tqdm import tqdm

from distill.data_util.prep_sst import SST


class IMDB(object):
  def __init__(self, data_path, pretrained=True):
    self.data_path = data_path
    self.eos = '<eos>'
    self.pad = '<pad>'
    self.unk = '<unk>'
    self.pretrained = pretrained
    self.load_vocab()


  def load_data(self):
    self.imdb = keras.datasets.imdb
    (train_data, train_labels), (test_data, test_labels) = self.imdb.load_data(num_words=10000)
    self.data = {}
    self.data['train'] = list(zip(train_data, train_labels))
    self.data['dev'] =  list(zip(test_data[:12000], test_labels[:12000]))
    self.data['test'] =  list(zip(test_data[12000:], test_labels[12000:]))

  def get_pretrained_path(self,pretrained_model):
    return os.path.join(self.data_path, "filtered_pretrained_"+pretrained_model)

  def get_pretrained_mat(self, pretrained_model):
    return np.load(self.get_pretrained_path(pretrained_model)+".npy")


  def prepare_pretrained(self, full_pretrained_path, pretrained_model, embedding_dim):
    filtered_path = self.get_pretrained_path(pretrained_model)
    full_embeddings = {}

    with open(full_pretrained_path, encoding='utf-8') as f:
      for line in f:
        line = line.strip()
        if not line: continue
        vocab, embed = line.split(u' ', 1)
        if vocab.lower() in self.word_to_index:
          full_embeddings[vocab] = np.asarray(embed.split(), dtype=np.float32)

    ordered_embeddings = []

    init_tokens = {'<pad>': np.random.uniform(
      -0.05, 0.05, embedding_dim).astype(np.float32),
                   '<eos>':np.random.uniform(
      -0.05, 0.05, embedding_dim).astype(np.float32),
                   '<unk>':np.random.uniform(
      -0.05, 0.05, embedding_dim).astype(np.float32)}

    print("total numbr of vocabs in filtered Glove", len(list(full_embeddings.keys())))
    print("total number of vocabs:", len(self.index_to_word))
    for key in np.arange(len(self.index_to_word)):
      token = self.index_to_word[key]
      if token in full_embeddings:
        ordered_embeddings.append(full_embeddings[token].astype(np.float32))
      elif token in init_tokens:
        ordered_embeddings.append(init_tokens[token])
      else:
        ordered_embeddings.append(init_tokens['<unk>'])


    np.save(filtered_path, ordered_embeddings, allow_pickle=True)

  def decode(self, ids):
    return [self.index_to_word[i] for i in ids]

  def encode(self, tokens):
    return [self.word_to_index[t] for t in tokens]

  def build_vocab(self):
    self.load_data()
    word2index = self.imdb.get_word_index()

    # The first indices are reserved
    word2index = {k: (v + 3) for k, v in word2index.items()}
    word2index["<pad>"] = 0
    word2index["<start>"] = 1
    word2index["<unk>"] = 2  # unknown
    word2index["<eos>"] = 3

    index2word = dict([(value, key) for (key, value) in word2index.items()])

    vocab_dict = {'word_to_id': word2index, 'id_to_word': index2word}
    np.save(os.path.join(self.data_path, "vocab"), vocab_dict, allow_pickle=True)

  def load_vocab(self):
    vocab_path = os.path.join(self.data_path, 'vocab.npy')

    if not os.path.exists(vocab_path):
      self.build_vocab()

    vocab = np.load(vocab_path, allow_pickle=True).item()
    self.word_to_index = vocab['word_to_id']
    self.index_to_word = vocab['id_to_word']

  def get_tf_example(self,example):
    """Convert our own representation of an example's features to Features class for TensorFlow dataset.
    """
    features = tf.train.Features(feature={
      "inputs": tf.train.Feature(int64_list=tf.train.Int64List(value=example['token_ids'])),
      "targets": tf.train.Feature(int64_list=tf.train.Int64List(value=example['labels'])),
      "inputs_length": tf.train.Feature(int64_list=tf.train.Int64List(value=[example['lengths']])),
    })
    return features

  def generator(self, mode):
    for example_id , (example) in enumerate(self.data[mode]):
      example = {
        'example_id': example_id,
        'labels': [example[1]],
        'lengths': len(example[0]),
        'token_ids': example[0],
      }

      yield example

  def build_tfrecords(self, mode):
    tf_example_features = []
    for example in self.generator(mode):
      tf_example_features.append(self.get_tf_example(example))

    with tf.python_io.TFRecordWriter(os.path.join(self.data_path,mode + ".tfr")) as tf_record_writer:
      for example in tqdm(tf_example_features):
        tf_record = tf.train.Example(features=example)
        tf_record_writer.write(tf_record.SerializeToString())

  @property
  def eos_id(self):
    return self.word_to_index[self.eos]

  @property
  def vocab_length(self):
    return len(self.index_to_word)

  @property
  def share_input_output_embeddings(self):
    return False

  @property
  def target_length(self):
    return 1

  @property
  def target_vocab(self):
    return [0,1]

  @staticmethod
  def get_padded_shapes():
    return [None], [None], [], []

  def get_tfrecord_path(self, mode, feature_type="full", add_subtrees=False):
    return os.path.join(self.data_path, mode + ".tfr")

  @staticmethod
  def parse_examples(example):
    """Load an example from TF record format."""
    features = {"inputs_length": tf.FixedLenFeature([], tf.int64),
                "targets": tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
                "inputs": tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
                }
    parsed_example = tf.parse_single_example(example, features=features)

    inputs_lengths = parsed_example["inputs_length"]
    targets_length = tf.ones(inputs_lengths.shape)
    inputs = parsed_example["inputs"]
    labels = parsed_example["targets"]

    return inputs, labels, inputs_lengths, targets_length


if __name__ == '__main__':
  imdb = IMDB('data/imdb')
  imdb.load_data()


  imdb.build_tfrecords("train")
  imdb.build_tfrecords("dev")
  imdb.build_tfrecords("test")

  print(sum(1 for _ in tf.python_io.tf_record_iterator(
    imdb.get_tfrecord_path(mode="train"))))
  print(sum(1 for _ in tf.python_io.tf_record_iterator(
    imdb.get_tfrecord_path(mode="test"))))
  print(sum(1 for _ in tf.python_io.tf_record_iterator(
    imdb.get_tfrecord_path(mode="dev"))))

  batch_size = 10
  dataset = tf.data.TFRecordDataset(imdb.get_tfrecord_path(mode="train"))
  dataset = dataset.map(imdb.parse_examples)
  dataset = dataset.padded_batch(batch_size, padded_shapes=imdb.get_padded_shapes())
  iterator = dataset.make_initializable_iterator()

  example = iterator.get_next()
  inputs, targets, inputs_length, targets_length = example

  global_step = tf.train.get_or_create_global_step()
  scaffold = tf.train.Scaffold(local_init_op=tf.group(tf.local_variables_initializer(),
                                                      iterator.initializer))
  with tf.train.MonitoredTrainingSession(checkpoint_dir='logs', scaffold=scaffold) as sess:
    inp, targ, tag_len = \
      sess.run([inputs, targets, targets_length])
    print(inp)
    print(targ)
    print(tag_len)
    print(imdb.decode(inp[0]))

  imdb.prepare_pretrained('data/glove.840B.300d.txt', 'glove_300', 300)
