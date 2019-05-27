import numpy as np
import tensorflow as tf
from random import choices, uniform, randint, sample
import os
from tqdm import tqdm

from distill.data_util.trees import Tree

def minus(x, y):
  return x - y

def plus(x, y):
  return x + y

def binary_math_tree_generator(length, numbers, ops, max_value):
  if length == 1:
    return str(np.random.choice(numbers))
  else:
    left_length = np.random.randint(1,length)
    right_length = length - left_length
    left_child_val = -1
    while left_child_val < 0 or left_child_val > max_value:
      left_child = binary_math_tree_generator(left_length, numbers, ops, max_value)
      left_child_val = eval(left_child)

    right_child_val = -1
    while right_child_val < 0 or right_child_val > max_value:
      right_child = binary_math_tree_generator(right_length, numbers, ops, max_value)
      right_child_val = eval(right_child)

    op = np.random.choice(ops)

    exp = ' '.join(['(',left_child,op,right_child,')'])
    #print(exp, '= ', eval(exp))
    return exp


def first_draft():
    max_length = 10
    min_length = 2

    numbers = [1,2,3,4,5,6,7,8,9,10]
    numbers_ids = [0,1,2,3,4,5,6,7,8,9]
    input_probs = [0.1]*10
    operations = [minus, plus]
    operations_codes = ['-','+']
    operations_ids = [0, 1]
    operation_probs = [0.5]*2

    length = randint(min_length, max_length)
    print("Length:", length)
    print("Random input:", choices(numbers, input_probs))
    print("Probability of adding open param: ", uniform(0,1))
    print("Random operation", choices(operations_codes, operation_probs))

    number_of_examples = 2000
    examples = []
    labels = []
    trees = []
    while len(examples) < number_of_examples:
      example = []
      open_params = 0
      length = randint(min_length, max_length)
      numb_ids = np.random.choice(numbers_ids, length, replace=True)
      op_ids = np.random.choice(operations_ids, length-1, replace=True)
      ops_codes = np.asarray(operations_codes)[op_ids]
      ops = np.asarray(operations)[op_ids]
      numbs = np.asarray(numbers)[numb_ids]
      for i in np.arange(length-1):
        opened = False
        if uniform(0,1) > 0.5: #open para?
          example.append('(')
          open_params +=1
          opened = True

        answer = 0
        example.append(str(numbs[i]))
        if not opened:
          while open_params > 0 and uniform(0,1) > 0.5: #close para?
            example.append(')')
            open_params -= 1

        example.append(ops_codes[i])

      example.append(str(numbs[length-1]))

      while open_params > 0:  # close all remaining open paras
        example.append(')')
        open_params -= 1


      if example not in examples:
        examples.append(example)
        labels.append(eval(' '.join(example)))
        trees.append(Tree(' '.join(example)))
        print(examples[-1], labels[-1], trees[-1].get_words())


class Arithmatic(object):
  def __init__(self, data_path):
    self.data_path = data_path
    self.task_name = 'arithmatic'
    self.vocab_path = os.path.join(self.data_path,'vocab')

    self.eos = '<eos>'
    self.pad = '<pad>'

    self.load_vocab()
    self.pretrained = False


  @property
  def share_input_output_embeddings(self):
    return True

  @property
  def vocab_length(self):
    return len(self.id2word)

  def load_vocab(self):
    self.id2word = [self.pad, self.eos] + list(map(str,np.arange(self.num_of_symbols))) + ['(',')','*','+','-']

    print(self.id2word)
    self.word2id = {}
    for i in np.arange(len(self.id2word)):
      print(i, self.id2word[i])
      self.word2id[self.id2word[i]] = i

  @property
  def eos_id(self):
    return self.word2id[self.eos]

  @property
  def target_vocab(self):
    return self.id2word

  def decode(self, ids):
    return [self.id2word[i] for i in ids]

  def encode(self, tokens):
    return [self.word2id[t] for t in tokens]


  @property
  def target_length(self):
    return 1

  @property
  def num_of_symbols(self):
      return 1001 #0-100

  @property
  def train_length(self):
    return 20

  @property
  def dev_length(self):
    return 80

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
    print("max_output_freq: ", max_output_freq)
    for i in tqdm(np.arange(number_of_examples)):
      exp = -1
      exp_str = '-1'
      while exp < 0 or exp >= self.num_of_symbols:
        length = np.random.randint(max_length) + 1
        exp_str = binary_math_tree_generator(length, np.arange(1,int(self.num_of_symbols/10)), ['-','-', '+', '+', '+', '*'], max_value)
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
                 'targets':self.encode(output),
                 'inputs_length': len(exp_tokens),
                 'targets_length': len(output)}

      yield example


  def build_tfrecords(self, number_of_examples, mode):
    tf_examples = []
    for example in self.generator(number_of_examples, mode):
       tf_examples.append(self.get_tf_example(example))

    with tf.python_io.TFRecordWriter(os.path.join(self.data_path, self.task_name + "_" + mode + ".tfr")) as tf_record_writer:
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


if __name__ == '__main__':
  bin_iden = Arithmatic('data/arithmatic')

  bin_iden.build_tfrecords(10000, 'train')
  bin_iden.build_tfrecords(2000, 'dev')
  bin_iden.build_tfrecords(2000, 'test')