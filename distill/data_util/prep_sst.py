import tensorflow as tf
import numpy as np
import os
import itertools
from collections import OrderedDict
from tqdm import tqdm
from distill.data_util.trees import Tree, leftTraverse, get_subtrees
from distill.data_util.vocab import Vocab, PretrainedVocab


def get_word_embs(word_emb_path, word_emb_size, vocabulary_size=99002):
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
      line = line.lstrip().rstrip().split(" ")
      word = line[0]
      vector = list(map(float, line[1:]))
      if word_emb_size != len(vector):
        raise Exception("Expected vector of size {}, but got vector of size {}.".format(word_emb_size, len(vector)))
      word_emb_matrix.append(vector)
      word2id[word] = idx
      idx += 1

  word_emb_matrix = np.array(word_emb_matrix, dtype=np.float32)
  print("Loaded word embedding matrix with shape {}.".format(word_emb_matrix.shape))


  return word_emb_matrix, word2id

class SST(object):
  def __init__(self, data_path, add_subtrees=False, pretrained=True, pretrained_path="data/sst/filtered_glove.txt", embedding_size=300):
    self.data_path = data_path
    self.add_subtrees = add_subtrees
    self.vocab_path = os.path.join(data_path, "pretrained_" if pretrained else '' +"vocab")
    self.eos = '<eos>'
    self.pad = '<pad>'

    if pretrained:
      self.vocab = PretrainedVocab(self.vocab_path, pretrained_path, embedding_size)
    else:
      self.vocab = Vocab(path=self.vocab_path)

    self.load_vocab()


  def decode(self, ids):
    return [self.vocab.index_to_word[i] for i in ids]

  def encode(self, tokens):
    return [self.vocab.word_to_index[t] for t in tokens]

  def load_vocab(self):
    if self.vocab.exists():
      self.vocab.load()
    else:
      self.load_data()

      # Get list of tokenized sentences
      train_sents = [t.get_words() for t in self.data["train"]]
      # Get list of all words
      all_words = list(itertools.chain.from_iterable(train_sents))
      self.vocab.build_vocab(all_words)
      self.vocab.save()

  def generator(self, mode):
    for example_id, tree in enumerate(self.data[mode]):
      words = tree.get_words()
      node = tree.root
      nodes_list = []
      leftTraverse(node, lambda node, args: args.append(node), nodes_list)
      node_to_index = OrderedDict()
      for i in range(len(nodes_list)):
        node_to_index[nodes_list[i]] = i
      example = {
        'example_id': example_id,
        'is_leaf': [int(node.isLeaf) for node in nodes_list],
        'left_children': [0] + [node_to_index[node.left]+1 if
                          not node.isLeaf else 0
                          for node in nodes_list],
        'right_children': [0] + [node_to_index[node.right]+1 if
                           not node.isLeaf else 0
                           for node in nodes_list],
        'node_word_ids': [0] + [self.vocab.encode(node.word)[0] if
                          node.word else 0
                          for node in nodes_list],
        'labels': [node.label for node in nodes_list],
        'binary_labels': [0 if node.label <= 2 else 1 for node in nodes_list],
        'length': len(nodes_list),
        'word_length': len(words),
        'word_ids': self.vocab.encode(words + [self.eos]),
        'root_label': [tree.root.label],
        'root_binary_label': [0 if tree.root.label <= 2 else 1]
      }

      yield example

  def get_all_tf_features(self, example_feaures):
    """Convert our own representation of an example's features to Features class for TensorFlow dataset.
    """
    features = tf.train.Features(feature={
      "example_id": tf.train.Feature(int64_list=tf.train.Int64List(value=[example_feaures['example_id']])),
      "is_leaf": tf.train.Feature(int64_list=tf.train.Int64List(value=example_feaures['is_leaf'])),
      "left_children": tf.train.Feature(int64_list=tf.train.Int64List(value=example_feaures['left_children'])),
      "right_children": tf.train.Feature(int64_list=tf.train.Int64List(value=example_feaures['right_children'])),
      "node_word_ids": tf.train.Feature(int64_list=tf.train.Int64List(value=example_feaures['node_word_ids'])),
      "labels": tf.train.Feature(int64_list=tf.train.Int64List(value=example_feaures['labels'])),
      "binary_labels": tf.train.Feature(int64_list=tf.train.Int64List(value=example_feaures['binary_labels'])),
      "length": tf.train.Feature(int64_list=tf.train.Int64List(value=[example_feaures['length']])),
      "root_label": tf.train.Feature(int64_list=tf.train.Int64List(value=example_feaures['root_label'])),
      "root_binary_label": tf.train.Feature(int64_list=tf.train.Int64List(value=example_feaures['root_binary_label'])),
      "word_length": tf.train.Feature(int64_list=tf.train.Int64List(value=[example_feaures['word_length']])),
      "word_ids": tf.train.Feature(int64_list=tf.train.Int64List(value=example_feaures['word_ids'])),

    })
    return features

  def get_plain_tf_features(self, example_feaures):
    """Convert our own representation of an example's features to Features class for TensorFlow dataset.
    """
    features = tf.train.Features(feature={
      "example_id": tf.train.Feature(int64_list=tf.train.Int64List(value=[example_feaures['example_id']])),
      "root_label": tf.train.Feature(int64_list=tf.train.Int64List(value=example_feaures['root_label'])),
      "root_binary_label": tf.train.Feature(int64_list=tf.train.Int64List(value=example_feaures['root_binary_label'])),
      "word_length": tf.train.Feature(int64_list=tf.train.Int64List(value=[example_feaures['length']])),
      "word_ids": tf.train.Feature(int64_list=tf.train.Int64List(value=[example_feaures['word_ids']])),

    })
    return features

  def load_data(self):
    data_splits = ["train", "test", "dev"]
    self.data = {}
    for tag in data_splits:
      file = os.path.join(self.data_path, tag + ".txt")
      print("Loading %s trees.." % file)
      with open(file, 'r') as fid:
        self.data[tag] = []
        for line in fid.readlines():
          tree = Tree(line)
          if self.add_subtrees and tag == 'train':
            sub_trees = get_subtrees(tree.root)
            self.data[tag].extend(sub_trees)
          else:
            self.data[tag].append(tree)

  def build_tfrecords(self,tf_feature_fn, mode, feature_type="tree"):
    tf_example_features = []
    for example in self.generator(mode):
      if example['root_label'] != 2:
       tf_example_features.append(tf_feature_fn(example))

    if mode == 'train':
      subtree_name_token = '_allsubs' if self.add_subtrees else ''
    else:
      subtree_name_token = ''

    with tf.python_io.TFRecordWriter(os.path.join(self.data_path,feature_type+"_" + mode + subtree_name_token + ".tfr")) as tf_record_writer:
      for example in tqdm(tf_example_features):
        tf_record = tf.train.Example(features=example)
        tf_record_writer.write(tf_record.SerializeToString())

  @property
  def eos_id(self):
    return self.vocab.word_to_index[self.eos]

  @property
  def vocab_length(self):
    return len(self.vocab.index_to_word)

  @property
  def share_input_output_embeddings(self):
    return False

  @property
  def target_length(self):
    return 1

  @property
  def target_vocab(self, fine_grained=True):
    if fine_grained:
      return [0,1,2,3,4]
    else:
      return [0, 1]

  @staticmethod
  def parse_sst_tree_examples(example):
    """Load an example from TF record format."""
    features = {"example_id": tf.FixedLenFeature([], tf.int64),
                "length": tf.FixedLenFeature([], tf.int64),
                "is_leaf": tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
                "left_children": tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
                "right_children": tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
                "node_word_ids": tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
                "labels": tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
                "binary_labels": tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True)}
    parsed_example = tf.parse_single_example(example, features=features)

    example_id = parsed_example["example_id"]
    length = parsed_example["length"]
    tf.logging.info(length)
    is_leaf = parsed_example["is_leaf"]
    tf.logging.info(is_leaf)
    left_children = parsed_example["left_children"]
    right_children = parsed_example["right_children"]
    node_word_ids = parsed_example["node_word_ids"]
    labels = parsed_example["labels"]
    binary_labels = parsed_example["binary_labels"]

    return example_id, length, is_leaf, left_children, right_children, node_word_ids, labels, binary_labels

  @staticmethod
  def parse_full_sst_tree_examples(example):
    """Load an example from TF record format."""
    features = {"example_id": tf.FixedLenFeature([], tf.int64),
                "length": tf.FixedLenFeature([], tf.int64),
                "is_leaf": tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
                "left_children": tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
                "right_children": tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
                "node_word_ids": tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
                "labels": tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
                "binary_labels": tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
                "root_label": tf.FixedLenFeature([], tf.int64),
                "root_binary_label": tf.FixedLenFeature([], tf.int64),
                "word_length": tf.FixedLenFeature([], tf.int64),
                "word_ids": tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
                }

    parsed_example = tf.parse_single_example(example, features=features)

    example_id = parsed_example["example_id"]
    length = parsed_example["length"]
    tf.logging.info(length)
    is_leaf = parsed_example["is_leaf"]
    tf.logging.info(is_leaf)
    left_children = parsed_example["left_children"]
    right_children = parsed_example["right_children"]
    node_word_ids = parsed_example["node_word_ids"]
    labels = parsed_example["labels"]
    binary_labels = parsed_example["binary_labels"]

    root_label = parsed_example["root_label"]
    root_binary_label = parsed_example["root_binary_label"]
    word_length = parsed_example["word_length"]
    word_ids = parsed_example["word_ids"]

    return example_id, length, is_leaf, left_children, right_children, node_word_ids, labels, binary_labels,\
           root_label, root_binary_label, word_length, word_ids

  @staticmethod
  def get_full_padded_shapes():
    return [None], [None], [], []

  @staticmethod
  def get_padded_shapes():
    return [None], [None], [], []

  def get_tfrecord_path(self, mode, feature_type="full", add_subtrees=True):

    if mode == "train":
      subtree_name_token = '_allsubs' if add_subtrees else ''
    else:
      subtree_name_token = ''
    return os.path.join(self.data_path, feature_type + "_" + mode + subtree_name_token + ".tfr")

  @staticmethod
  def parse_examples(example):
    """Load an example from TF record format."""
    features = {"word_length": tf.FixedLenFeature([], tf.int64),
                "root_label": tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
                "word_ids": tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
                }
    parsed_example = tf.parse_single_example(example, features=features)

    inputs_lengths = parsed_example["word_length"]
    targets_length = tf.ones(inputs_lengths.shape)
    inputs = parsed_example["word_ids"]
    labels = parsed_example["root_label"]

    return inputs, labels, inputs_lengths, targets_length

def build_sst():

  sst_prep = SST(data_path="data/sst/")
  sst_prep.load_data()
  sst_prep.build_tfrecords("train")
  sst_prep.build_tfrecords("dev")
  sst_prep.build_tfrecords("test")

def build_full_sst():
  sst_prep = SST(data_path="data/sst/",
                 add_subtrees=True,
                 pretrained=True,
                 pretrained_path="data/sst/filtered_glove.txt",
                 embedding_size=300)
  sst_prep.load_data()

  sst_prep.build_tfrecords(sst_prep.get_all_tf_features, mode="train", feature_type="full")
  sst_prep.build_tfrecords(sst_prep.get_all_tf_features, mode="dev", feature_type="full")
  sst_prep.build_tfrecords(sst_prep.get_all_tf_features, mode="test", feature_type="full")

def build_sst_main():
  build_sst()

  dataset = tf.data.TFRecordDataset(SST.get_tfrecord_path("data/sst", mode="train"))
  dataset = dataset.map(SST.parse_sst_tree_examples)
  dataset = dataset.padded_batch(10, padded_shapes=SST.get_padded_shapes())
  iterator = dataset.make_initializable_iterator()

  example = iterator.get_next()
  example_id, length, is_leaf, left_children, right_children, node_word_ids, labels, binary_labels = example
  global_step = tf.train.get_or_create_global_step()
  scaffold = tf.train.Scaffold(local_init_op=tf.group(tf.local_variables_initializer(),
                                                      iterator.initializer))
  with tf.train.MonitoredTrainingSession(checkpoint_dir='logs', scaffold=scaffold) as sess:
    print(sess.run(labels))

def test_seq2seq():
  sst_prep = SST(data_path="data/sst/",
                 add_subtrees=True,
                 pretrained=True,
                 pretrained_path="data/sst/filtered_glove.txt",
                 embedding_size=300)

  batch_size = 10
  dataset = tf.data.TFRecordDataset(sst_prep.get_tfrecord_path(mode="train", feature_type="full"))
  dataset = dataset.map(sst_prep.parse_examples)
  dataset = dataset.padded_batch(batch_size, padded_shapes=sst_prep.get_padded_shapes())
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



def test():
  batch_size = 10
  dataset = tf.data.TFRecordDataset(SST.get_tfrecord_path("data/sst", mode="train", feature_type="full"))
  dataset = dataset.map(SST.parse_full_sst_tree_examples)
  dataset = dataset.padded_batch(batch_size, padded_shapes=SST.get_full_padded_shapes())
  iterator = dataset.make_initializable_iterator()

  example = iterator.get_next()
  example_id, length, is_leaf, left_children, right_children, node_word_ids, labels, binary_labels, \
      root_label, root_binary_label, word_length, word_ids = example

  bach_indices = tf.expand_dims(tf.range(batch_size), 1)
  root_indices = tf.concat([bach_indices, tf.expand_dims(tf.cast(length - 1, dtype=tf.int32), 1)], axis=-1)

  computed_root_label = tf.gather_nd(labels, root_indices)
  computed_root_binary_label = tf.gather_nd(binary_labels, root_indices)

  global_step = tf.train.get_or_create_global_step()
  scaffold = tf.train.Scaffold(local_init_op=tf.group(tf.local_variables_initializer(),
                                                      iterator.initializer))
  with tf.train.MonitoredTrainingSession(checkpoint_dir='logs', scaffold=scaffold) as sess:
    out_root_label, out_root_binary_label, out_labels, out_binary_labels, out_computed_root_label, out_computed_root_binary_label= \
      sess.run([root_label, root_binary_label,labels, binary_labels, computed_root_label, computed_root_binary_label])
    out_word_ids = sess.run(word_ids)

    print(out_word_ids)

    print(list(zip(out_root_label,out_computed_root_label)))
    print(out_root_binary_label[0])
    print(out_computed_root_label[0])
    print(out_computed_root_binary_label[0])

if __name__ == '__main__':
  build_full_sst()
  #test_seq2seq()

  sst_prep = SST(data_path="data/sst/",
                 add_subtrees=True,
                 pretrained=True,
                 pretrained_path="data/sst/filtered_glove.txt",
                 embedding_size=300)

  print(sum(1 for _ in tf.python_io.tf_record_iterator(sst_prep.get_tfrecord_path(mode="train", feature_type="full", add_subtrees=True))))
  print(sum(1 for _ in tf.python_io.tf_record_iterator(sst_prep.get_tfrecord_path(mode="test", feature_type="full", add_subtrees=True))))
  print(sum(1 for _ in tf.python_io.tf_record_iterator(sst_prep.get_tfrecord_path(mode="dev", feature_type="full", add_subtrees=True))))

