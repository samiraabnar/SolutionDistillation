import numpy as np
import tensorflow as tf
from tqdm import tqdm
import os


class Algorithmic(object):
  def __init__(self, data_path):
    self.data_path = data_path
    self.task_name = 'algorithmic'


  def get_tf_example(self, example):
    """Convert our own representation of an example's features to Features class for TensorFlow dataset.
    """
    features = tf.train.Features(feature={
      "inputs": tf.train.Feature(int64_list=tf.train.Int64List(value=example['inputs'])),
      "targets": tf.train.Feature(int64_list=tf.train.Int64List(value=example['targets'])),
      "length": tf.train.Feature(int64_list=tf.train.Int64List(value=[example['length']]))
    })
    return features

  def generator(self, number_of_examples):
    raise NotImplementedError()

  def build_tfrecords(self, number_of_examples, mode):
    tf_examples = []
    for example in self.generator(number_of_examples):
       tf_examples.append(self.get_tf_example(example))

    with tf.python_io.TFRecordWriter(os.path.join(self.data_path, self.task_name+"_"+mode + ".tfr")) as tf_record_writer:
      for example in tqdm(tf_examples):
        tf_record = tf.train.Example(features=example)
        tf_record_writer.write(tf_record.SerializeToString())

  @staticmethod
  def parse_examples(example):
    """Load an example from TF record format."""
    features = {"length": tf.FixedLenFeature([], tf.int64),
                "targets": tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
                "inputs": tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
                }
    parsed_example = tf.parse_single_example(example, features=features)

    lengths = parsed_example["length"]
    inputs = parsed_example["inputs"]
    labels = parsed_example["targets"]

    return inputs, labels, lengths

  @staticmethod
  def get_padded_shapes():
    return [None], [None], []

  def get_tfrecord_path(self, mode):
    return os.path.join(self.data_path, self.task_name +"_"+mode + ".tfr")


class AlgorithmicIdentityBinary40(Algorithmic):
  """Problem spec for algorithmic binary identity task."""

  def __init__(self, data_path):
    super(AlgorithmicIdentityBinary40, self).__init__(data_path=data_path)
    self.task_name = 'identity_binary_40'

  @property
  def num_symbols(self):
    return 2

  @property
  def max_length(self):
    return 40

  def generator(self, number_of_examples):
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
    for _ in range(number_of_examples):
      l = np.random.randint(self.max_length) + 1
      inputs = [np.random.randint(self.num_symbols) for _ in range(l)]

      yield {"inputs": inputs, "targets": inputs, 'length':l}


if __name__ == '__main__':
    bin_iden = AlgorithmicIdentityBinary40('data/alg')
    bin_iden.build_tfrecords(5000, 'train')
    bin_iden.build_tfrecords(1000, 'dev')
    bin_iden.build_tfrecords(1000, 'test')



    dataset = tf.data.TFRecordDataset(bin_iden.get_tfrecord_path(mode="train"))
    dataset = dataset.map(bin_iden.parse_examples)
    dataset = dataset.padded_batch(1, padded_shapes=bin_iden.get_padded_shapes())
    iterator = dataset.make_initializable_iterator()

    example = iterator.get_next()
    inputs, labels, lengths = example
    global_step = tf.train.get_or_create_global_step()
    scaffold = tf.train.Scaffold(local_init_op=tf.group(tf.local_variables_initializer(),
                                                        iterator.initializer))
    with tf.train.MonitoredTrainingSession(checkpoint_dir='logs', scaffold=scaffold) as sess:
      print(sess.run([inputs, labels, lengths]))

