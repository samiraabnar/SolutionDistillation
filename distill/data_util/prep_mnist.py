import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm
import tensorflow_datasets as tfds

class Mnist1D(object):
  def __init__(self, data_path):
    self.data_path = data_path
    self.task_name = 'mnist_1d'
    self.vocab_path = os.path.join(self.data_path, 'vocab')

    self.eos = '<eos>'
    self.pad = '<pad>'
    self.cls_token = '<cls>'

    self.load_vocab()
    self.pretrained = False

  @property
  def if_label_gaussian_noise(self):
    return False

  @property
  def guassian_noise_scale(self):
    return 0.9

  @property
  def share_input_output_embeddings(self):
    return True

  @property
  def vocab_length(self):
    return len(self.id2word)

  def load_vocab(self):
    raise NotImplementedError

  @property
  def eos_id(self):
    return self.word2id[self.eos]

  @property
  def target_vocab(self):
    return self.id2word  # list(np.arange(self.num_of_symbols))

  def decode(self, ids):
    return [self.id2word[i] for i in ids]

  def encode(self, tokens):
    return [self.word2id[t] for t in tokens]

  @property
  def target_length(self):
    return 1

  @property
  def num_of_symbols(self):
    raise NotImplementedError

  def get_tf_example(self, example):
    """Convert our own representation of an example's features to Features class for TensorFlow dataset.
    """
    features = tf.train.Features(feature={
      "inputs": tf.train.Feature(int64_list=tf.train.Int64List(value=example['inputs'])),
      "targets": tf.train.Feature(int64_list=tf.train.Int64List(value=example['targets'])),
      "inputs_length": tf.train.Feature(int64_list=tf.train.Int64List(value=[example['inputs_length']])),
      "targets_length": tf.train.Feature(int64_list=tf.train.Int64List(value=[example['targets_length']]))

    })
    return features

  def generator(self, number_of_examples, mode="train"):
    max_length = self.train_length if mode == "train" else self.dev_length
    budgets = {}
    max_value = self.num_of_symbols - 1
    max_output_freq = (number_of_examples / self.num_of_symbols) * 2
    for i in tqdm(np.arange(number_of_examples)):
      exp = -1
      exp_str = '-1'
      while exp < 0 or exp >= self.num_of_symbols:
        length = np.random.randint(max_length) + 1
        exp_str, _ = binary_math_tree_generator(length, np.arange(1, int(self.num_of_symbols)),
                                                ['-', '-', '+', '+', '+', '*'], max_value, 0, self.max_depth)
        exp = eval(exp_str)
        if exp not in budgets:
          budgets[exp] = 1
        budgets[exp] += 1
        if budgets[exp] >= max_output_freq:
          exp = -1
          exp_str = '-1'

      exp_tokens = exp_str.split() + [self.eos]
      output = [str(exp)]
      example = {'inputs': self.encode(exp_tokens),
                 'targets': self.encode(output),
                 'inputs_length': len(exp_tokens),
                 'targets_length': len(output)}

      yield example

  def build_tfrecords(self, number_of_examples, mode):
    tf_examples = []
    for example in self.generator(number_of_examples, mode):
      tf_examples.append(self.get_tf_example(example))

    with tf.python_io.TFRecordWriter(
        os.path.join(self.data_path, self.task_name + "_" + mode + ".tfr")) as tf_record_writer:
      for example in tqdm(tf_examples):
        tf_record = tf.train.Example(features=example)
        tf_record_writer.write(tf_record.SerializeToString())

  @staticmethod
  def parse_examples(example):
    """Load an example from TF record format."""
    features = {"inputs_length": tf.FixedLenFeature([], tf.int64),
                "targets_length": tf.FixedLenFeature([], tf.int64),
                "targets": tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
                "inputs": tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
                }
    parsed_example = tf.parse_single_example(example, features=features)

    inputs_lengths = parsed_example["inputs_length"]
    targets_length = parsed_example["targets_length"]
    inputs = parsed_example["inputs"]
    labels = parsed_example["targets"]

    return inputs, labels, inputs_lengths, targets_length

  @staticmethod
  def get_padded_shapes():
    return [None], [None], [], []

  def get_tfrecord_path(self, mode):
    return os.path.join(self.data_path, self.task_name + "_" + mode + ".tfr")


if __name__ == '__main__':
  mnist_train = tfds.load(name="mnist", split="train")
  assert isinstance(mnist_train, tf.data.Dataset)
  print(mnist_train)