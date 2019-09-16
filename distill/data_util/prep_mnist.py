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
    return 28*28

  def load_vocab(self):
    raise NotImplementedError

  @property
  def eos_id(self):
    return None

  @property
  def target_vocab(self):
    return [0,1,2,3,4,5,6,7,8,9]  # list(np.arange(self.num_of_symbols))

  def decode(self, ids):
    return ids

  def encode(self, tokens):
    return tokens

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

  def generator(self, number_of_examples, split):
    train_ds = tfds.load("mnist", split=split, batch_size=-1)
    numpy_ds = tfds.as_numpy(train_ds)
    numpy_images, numpy_labels = numpy_ds["image"], numpy_ds["label"]


    _, height, width, _ = numpy_images.shape
    for image, label in zip(numpy_images, numpy_labels):
      inputs = image.reshape((height*width))
      targets = [label]
      example = {'inputs': inputs,
                 'targets': targets,
                 'inputs_length': len(inputs),
                 'targets_length': len(targets)}

      yield example

  def build_tfrecords(self, number_of_examples, mode):
    tf_examples = []
    for example in self.generator(number_of_examples, mode):
      tf_examples.append(self.get_tf_example(example))

    with tf.python_io.TFRecordWriter(
        os.path.join(self.data_path, self.task_name + "_" + str(mode) + ".tfr")) as tf_record_writer:
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
  mnist_builder = Mnist1D("data/mnist1d/")
  mnist_builder.build_tfrecords(None, tfds.Split.TRAIN)
  mnist_builder.build_tfrecords(None, tfds.Split.TEST)
  mnist_builder.build_tfrecords(None, tfds.Split.VALIDATION)
