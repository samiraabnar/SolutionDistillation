import numpy as np
import tensorflow as tf
from tqdm import tqdm
import os

from distill.common.util import random_number_lower_endian, lower_endian_to_number, number_to_lower_endian


class Algorithmic(object):
  def __init__(self, data_path):
    self.data_path = data_path
    self.task_name = 'algorithmic'
    self.eos = '<eos>'


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
    raise NotImplementedError()

  def build_tfrecords(self, number_of_examples, mode):
    tf_examples = []
    for example in self.generator(number_of_examples, mode):
       tf_examples.append(self.get_tf_example(example))

    with tf.python_io.TFRecordWriter(os.path.join(self.data_path, self.task_name+"_"+mode + ".tfr")) as tf_record_writer:
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
    return os.path.join(self.data_path, self.task_name +"_"+mode + ".tfr")


class AlgorithmicIdentityDecimal40(Algorithmic):
  """Problem spec for algorithmic decimal identity task."""

  def __init__(self, data_path):
    super(AlgorithmicIdentityDecimal40, self).__init__(data_path=data_path)
    self.task_name = 'identity_decimal_40'

  @property
  def num_symbols(self):
    return 10

  @property
  def train_length(self):
    return 40

  @property
  def dev_length(self):
    return 400


  def generator(self, number_of_examples, mode="train"):
    """Generator for the identity (copy) task on sequences of symbols.
    The length of the sequence is drawn uniformly at random from [1, max_length]
    and then symbols are drawn uniformly at random from [0, nbr_symbols) until
    nbr_cases sequences have been produced.
    Args:
      nbr_symbols: number of symbols to use in each sequence.
      max_length: integer, maximum length of sequences to generate.
      nbr_cases: the number of cases to generate.
    Yields:
      A dictionary {"inputs": input-list, "targets": target-list} where
      input-list and target-list are the same.
    """

    max_length = self.train_length if mode=="train" else self.dev_length
    for _ in range(number_of_examples):
      l = np.random.randint(max_length) + 1
      inputs = [np.random.randint(self.num_symbols) for _ in range(l)]

      yield {"inputs": inputs, "targets": inputs, 'inputs_length':l, "targets_length": len(inputs)}

class AlgorithmicIdentityBinary40(Algorithmic):
  """Problem spec for algorithmic decimal identity task."""

  def __init__(self, data_path):
    super(AlgorithmicIdentityBinary40, self).__init__(data_path=data_path)
    self.task_name = 'identity_binary_40'
    self.load_vocab()

  @property
  def num_symbols(self):
    return 2

  @property
  def train_length(self):
    return 40

  @property
  def dev_length(self):
    return 400


  @property
  def vocab_length(self):
    return len(self.id2word)

  @property
  def eos_id(self):
    return self.word2id[self.eos]

  def load_vocab(self):
    self.id2word = [i for i in np.arange(self.num_symbols)] + [self.eos]

    self.word2id = {}
    for i,word in enumerate(self.id2word):
      self.word2id[word] = i

  def encode(self, tokens):
    return [self.word2id[t] for t in tokens]


  def decode(self, ids):
    return [self.id2word[i] for i in ids]

  def generator(self, number_of_examples, mode="train"):
    """Generator for the identity (copy) task on sequences of symbols.
    The length of the sequence is drawn uniformly at random from [1, max_length]
    and then symbols are drawn uniformly at random from [0, nbr_symbols) until
    nbr_cases sequences have been produced.
    Args:
      nbr_symbols: number of symbols to use in each sequence.
      max_length: integer, maximum length of sequences to generate.
      nbr_cases: the number of cases to generate.
    Yields:
      A dictionary {"inputs": input-list, "targets": target-list} where
      input-list and target-list are the same.
    """

    max_length = self.train_length if mode=="train" else self.dev_length
    for _ in range(number_of_examples):
      l = np.random.randint(max_length) + 1
      inputs = self.encode([np.random.randint(self.num_symbols) for _ in range(l)] + [self.eos])

      yield {"inputs": inputs, "targets": inputs, 'inputs_length':len(inputs), "targets_length": len(inputs)}


class AlgorithmicAdditionDecimal40(Algorithmic):
  """Problem spec for algorithmic decimal addition task."""

  def __init__(self, data_path):
    super(AlgorithmicAdditionDecimal40, self).__init__(data_path=data_path)
    self.task_name = 'addition_decimal_40'

  @property
  def num_symbols(self):
    return 10

  @property
  def train_length(self):
    return 40

  @property
  def dev_length(self):
    return 400

  @property
  def base(self):
    return 10

  def generator(self, number_of_examples, mode="train"):  # pylint: disable=arguments-differ
    """Generator for the addition task.
    The length of each number is drawn uniformly at random in [1, max_length/2]
    and then digits are drawn uniformly at random. The numbers are added and
    separated by [base] in the input. Stops at nbr_cases.
    Args:
      base: in which base are the numbers.
      max_length: integer, maximum length of sequences to generate.
      nbr_cases: the number of cases to generate.
    Yields:
      A dictionary {"inputs": input-list, "targets": target-list} where
      input-list are the 2 numbers and target-list is the result of adding them.
    Raises:
      ValueError: if max_length is lower than 3.
    """
    max_length = self.train_length if mode=="train" else self.dev_length

    if max_length < 3:
      raise ValueError("Maximum length must be at least 3.")
    for _ in range(number_of_examples):
      l1 = np.random.randint(max_length // 2) + 1
      l2 = np.random.randint(max_length - l1 - 1) + 1
      n1 = random_number_lower_endian(l1, self.base)
      n2 = random_number_lower_endian(l2, self.base)
      result = lower_endian_to_number(n1, self.base) + lower_endian_to_number(
          n2, self.base)
      inputs = n1 + [self.base] + n2
      targets = number_to_lower_endian(result, self.base)
      yield {"inputs": inputs, "targets": targets, "inputs_length": l1+l2+1, "targets_length": len(targets)}


class AlgorithmicMultiplicationDecimal40(Algorithmic):
  """Problem spec for algorithmic decimal multiplication task."""
  def __init__(self, data_path):
    super(AlgorithmicMultiplicationDecimal40, self).__init__(data_path=data_path)
    self.task_name = 'multiplication_decimal_40'

  @property
  def num_symbols(self):
    return 10

  @property
  def train_length(self):
    return 40

  @property
  def dev_length(self):
    return 400

  @property
  def base(self):
    return 10

  def generator(self, number_of_examples, mode="train"):  # pylint: disable=arguments-differ
    """Generator for the multiplication task.
    The length of each number is drawn uniformly at random in [1, max_length/2]
    and then digits are drawn uniformly at random. The numbers are multiplied
    and separated by [base] in the input. Stops at nbr_cases.
    Args:
      base: in which base are the numbers.
      max_length: integer, maximum length of sequences to generate.
      nbr_cases: the number of cases to generate.
    Yields:
      A dictionary {"inputs": input-list, "targets": target-list} where
      input-list are the 2 numbers and target-list is the result of multiplying
      them.
    Raises:
      ValueError: if max_length is lower than 3.
    """
    max_length = self.train_length if mode=="train" else self.dev_length

    if max_length < 3:
      raise ValueError("Maximum length must be at least 3.")
    for _ in range(number_of_examples):
      l1 = np.random.randint(max_length // 2) + 1
      l2 = np.random.randint(max_length - l1 - 1) + 1
      n1 = random_number_lower_endian(l1, self.base)
      n2 = random_number_lower_endian(l2, self.base)
      result = lower_endian_to_number(n1, self.base) * lower_endian_to_number(
          n2, self.base)
      inputs = n1 + [self.base] + n2
      targets = number_to_lower_endian(result, self.base)
      yield {"inputs": inputs, "targets": targets, 'inputs_length': len(inputs), "targets_length": len(targets)}


class AlgorithmicReverseProblem(Algorithmic):
  """Problem spec for sorting numbers."""
  def __init__(self, data_path):
    super(AlgorithmicReverseProblem, self).__init__(data_path=data_path)
    self.task_name = 'sort'

  @property
  def num_symbols(self):
    return max(self.train_length, self.dev_length)

  @property
  def train_length(self):
    return 10

  @property
  def dev_length(self):
    return self.train_length * 2

  @property
  def unique(self):
    """Unique numbers wo/ replacement or w/ replacement in sorting task."""
    return False

  def generator(self, number_of_examples, mode='train'):
    """Generating for sorting task on sequence of symbols.
    The length of the sequence is drawn uniformly at random from [1, max_length]
    and then symbols are drawn (uniquely w/ or w/o replacement) uniformly at
    random from [0, nbr_symbols) until nbr_cases sequences have been produced.
    Args:
      nbr_symbols: number of symbols to use in each sequence.
      max_length: integer, maximum length of sequences to generate.
      nbr_cases: the number of cases to generate.
    Yields:
      A dictionary {"inputs": input-list, "targets": target-list} where
      target-list is input-list sorted.
    """
    if mode == 'train':
      max_length = self.train_length
    else:
      max_length = self.dev_length

    for _ in range(number_of_examples):
      # Sample the sequence length.
      length = np.random.randint(max_length) + 1

      if self.unique:
        # Sample our inputs w/o replacement.
        inputs = np.arange(self.num_symbols)
        np.random.shuffle(inputs)

        # Truncate to the desired length.
        inputs = inputs[:length]
        inputs = list(inputs)
      else:
        inputs = list(np.random.randint(self.num_symbols, size=length))

      # Targets are simply the sorted inputs.
      targets = list(reversed(inputs))

      yield {"inputs": inputs, "targets": targets, "inputs_length": len(inputs), "targets_length": len(targets)}


class AlgorithmicSortProblem(Algorithmic):
  """Problem spec for sorting numbers."""
  def __init__(self, data_path):
    super(AlgorithmicSortProblem, self).__init__(data_path=data_path)
    self.task_name = 'sort'

  @property
  def num_symbols(self):
    return max(self.train_length, self.dev_length)

  @property
  def train_length(self):
    return 10

  @property
  def dev_length(self):
    return self.train_length * 2

  @property
  def unique(self):
    """Unique numbers wo/ replacement or w/ replacement in sorting task."""
    return False

  def generator(self, number_of_examples, mode='train'):
    """Generating for sorting task on sequence of symbols.
    The length of the sequence is drawn uniformly at random from [1, max_length]
    and then symbols are drawn (uniquely w/ or w/o replacement) uniformly at
    random from [0, nbr_symbols) until nbr_cases sequences have been produced.
    Args:
      nbr_symbols: number of symbols to use in each sequence.
      max_length: integer, maximum length of sequences to generate.
      nbr_cases: the number of cases to generate.
    Yields:
      A dictionary {"inputs": input-list, "targets": target-list} where
      target-list is input-list sorted.
    """
    if mode == 'train':
      max_length = self.train_length
    else:
      max_length = self.dev_length

    for _ in range(number_of_examples):
      # Sample the sequence length.
      length = np.random.randint(max_length) + 1

      if self.unique:
        # Sample our inputs w/o replacement.
        inputs = np.arange(self.num_symbols)
        np.random.shuffle(inputs)

        # Truncate to the desired length.
        inputs = inputs[:length]
        inputs = list(inputs)
      else:
        inputs = list(np.random.randint(self.num_symbols, size=length))

      # Targets are simply the sorted inputs.
      targets = list(sorted(inputs))

      yield {"inputs": inputs, "targets": targets, "inputs_length": len(inputs), 'targets_length': len(targets)}

if __name__ == '__main__':
    """
    bin_iden = AlgorithmicSortProblem('data/alg')
    bin_iden.build_tfrecords(100000, 'train')
    bin_iden.build_tfrecords(10000, 'dev')
    bin_iden.build_tfrecords(10000, 'test')

    bin_iden = AlgorithmicReverseProblem('data/alg')
    bin_iden.build_tfrecords(100000, 'train')
    bin_iden.build_tfrecords(10000, 'dev')
    bin_iden.build_tfrecords(10000, 'test')

    bin_iden = AlgorithmicMultiplicationDecimal40('data/alg')
    bin_iden.build_tfrecords(100000, 'train')
    bin_iden.build_tfrecords(10000, 'dev')
    bin_iden.build_tfrecords(10000, 'test')

    bin_iden = AlgorithmicAdditionDecimal40('data/alg')
    bin_iden.build_tfrecords(100000, 'train')
    bin_iden.build_tfrecords(10000, 'dev')
    bin_iden.build_tfrecords(10000, 'test')

    bin_iden = AlgorithmicIdentityDecimal40('data/alg')
    bin_iden.build_tfrecords(100000, 'train')
    bin_iden.build_tfrecords(10000, 'dev')
    bin_iden.build_tfrecords(10000, 'test')
    """

    bin_iden = AlgorithmicIdentityBinary40('data/alg')
    bin_iden.build_tfrecords(100000, 'train')
    bin_iden.build_tfrecords(10000, 'dev')
    bin_iden.build_tfrecords(10000, 'test')


    dataset = tf.data.TFRecordDataset(bin_iden.get_tfrecord_path(mode="train"))
    dataset = dataset.map(bin_iden.parse_examples)
    dataset = dataset.padded_batch(1, padded_shapes=bin_iden.get_padded_shapes())
    iterator = dataset.make_initializable_iterator()

    example = iterator.get_next()
    inputs, labels, inputs_lengths, targets_lengths = example
    global_step = tf.train.get_or_create_global_step()
    scaffold = tf.train.Scaffold(local_init_op=tf.group(tf.local_variables_initializer(),
                                                        iterator.initializer))
    with tf.train.MonitoredTrainingSession(checkpoint_dir='logs', scaffold=scaffold) as sess:
      print(sess.run([inputs, labels, inputs_lengths]))

