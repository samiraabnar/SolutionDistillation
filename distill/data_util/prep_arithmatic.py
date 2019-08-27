import numpy as np
import tensorflow as tf
from random import choices, uniform, randint, sample
import os
from tqdm import tqdm
import scipy.stats as stats
from num2words import num2words

from distill.data_util.trees import Tree

def binary_math_tree_generator(length, numbers, ops, max_value, depth=0, max_depth=None):
  if max_depth == None:
      max_depth = length
  if length == 1:
    return str(np.random.choice(numbers)), 0
  else:
    left_length = np.random.randint(1,length)
    right_length = length - left_length
    left_child_val = -1
    while left_child_val < 0 or left_child_val > max_value:
      left_child, depth_left = binary_math_tree_generator(left_length, numbers, ops, max_value, depth, max_depth)
      left_child_val = eval(left_child)

    right_child_val = -1
    while right_child_val < 0 or right_child_val > max_value:
      right_child, depth_right = binary_math_tree_generator(right_length, numbers, ops, max_value, depth, max_depth)
      right_child_val = eval(right_child)

    op = np.random.choice(ops)

    depth = max([depth_left, depth_right])
    add_paranthesis = np.random.choice([True, False])
    if add_paranthesis == True and depth < max_depth:
      exp = ' '.join(['(',left_child,op,right_child,')'])
      depth += 1
    else:
      exp = ' '.join([left_child, op, right_child])

    #print(exp, '= ', eval(exp))
    return exp, depth


def eq2str(eq, op_dic={'-':'minus', '+':'plus', '(':'(', ')':')', '*':'multiply'}):
  translation = []
  for item in eq:
    if item in op_dic:
      translation.append(op_dic[item])
    else:
      translation.append(num2words(item))
  return translation


class Arithmatic(object):
  def __init__(self, data_path):
    self.data_path = data_path
    self.task_name = 'arithmatic'
    self.vocab_path = os.path.join(self.data_path,'vocab')

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
    self.id2word = [self.pad, self.eos, self.cls_token] + list(map(str,np.arange(self.num_of_symbols))) + ['(',')','*','+','-']

    self.word2id = {}
    for i in np.arange(len(self.id2word)):
      self.word2id[self.id2word[i]] = i

  @property
  def eos_id(self):
    return self.word2id[self.eos]

  @property
  def target_vocab(self):
    return self.id2word #list(np.arange(self.num_of_symbols))

  def decode(self, ids):
    return [self.id2word[i] for i in ids]

  def encode(self, tokens):
    return [self.word2id[t] for t in tokens]


  @property
  def target_length(self):
    return 1

  @property
  def num_of_symbols(self):
      return 202 #0-100

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
    for i in tqdm(np.arange(number_of_examples)):
      exp = -1
      exp_str = '-1'
      while exp < 0 or exp >= self.num_of_symbols:
        length = np.random.randint(max_length) + 1
        exp_str, _ = binary_math_tree_generator(length, np.arange(1,int(self.num_of_symbols)), ['-','-', '+', '+', '+', '*'], max_value, 0, self.max_depth)
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


class ArithmaticSameLength(Arithmatic):
  def __init__(self, data_path):
    super(ArithmaticSameLength, self).__init__(data_path)
    self.task_name = 'arithmatic_samelen'


    self.load_vocab()
    self.pretrained = False

  @property 
  def max_depth(self):
    return 40

  @property
  def train_length(self):
    return 40

  @property
  def dev_length(self):
    return 40


class ArithmaticSimple(ArithmaticSameLength):
  def __init__(self, data_path):
    self.data_path = data_path
    self.task_name = 'arithmatic_simple'
    self.vocab_path = os.path.join(self.data_path,'vocab')

    self.eos = '<eos>'
    self.pad = '<pad>'
    self.cls_token = '<cls>'

    self.load_vocab()
    self.pretrained = False

  def load_vocab(self):
    self.id2word = [self.pad, self.eos, self.cls_token] + list(map(str,np.arange(self.num_of_symbols))) + ['(',')','+','-']

    self.word2id = {}
    for i in np.arange(len(self.id2word)):
      self.word2id[self.id2word[i]] = i

  @property
  def eos_id(self):
    return self.word2id[self.eos]

  @property
  def num_of_symbols(self):
      return 202 #0-100

  @property 
  def max_depth(self):
    return 80

  @property
  def train_length(self):
    return 20

  @property
  def dev_length(self):
    return 80

  def generator(self, number_of_examples, mode="train"):
    max_length = self.train_length if mode == "train" else self.dev_length
    budgets = {}
    max_value = self.num_of_symbols - 1
    max_output_freq = (number_of_examples / self.num_of_symbols) * 2
    for i in tqdm(np.arange(number_of_examples)):
      exp = -1
      exp_str = '-1'
      while exp <= 0 or exp >= self.num_of_symbols:
        length = np.random.randint(max_length) + 1
        exp_str, _ = binary_math_tree_generator(length, np.arange(1,int(self.num_of_symbols)), ['-','+'], max_value, 0, self.max_depth)
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

class ArithmaticSimpleSameLength(ArithmaticSimple):
  def __init__(self, data_path):
    self.data_path = data_path
    self.task_name = 'arithmatic_simple'
    self.vocab_path = os.path.join(self.data_path,'vocab')

    self.eos = '<eos>'
    self.pad = '<pad>'
    self.cls_token = '<cls>'

    self.load_vocab()
    self.pretrained = False

  def load_vocab(self):
    self.id2word = [self.pad, self.eos, self.cls_token] + list(map(str,np.arange(self.num_of_symbols))) + ['(',')','+','-']

    self.word2id = {}
    for i in np.arange(len(self.id2word)):
      self.word2id[self.id2word[i]] = i

  @property
  def eos_id(self):
    return self.word2id[self.eos]

  @property
  def num_of_symbols(self):
      return 202 #0-100

  @property 
  def max_depth(self):
    return 40
    
  @property
  def train_length(self):
    return 40

  @property
  def dev_length(self):
    return 40
  
class ArithmaticSimpleCurriculumLength(ArithmaticSimple):
  def __init__(self, data_path):
    self.data_path = data_path
    self.task_name = 'arithmatic_simple'
    self.vocab_path = os.path.join(self.data_path,'vocab')

    self.eos = '<eos>'
    self.pad = '<pad>'
    self.cls_token = '<cls>'

    self.load_vocab()
    self.pretrained = False

  def load_vocab(self):
    self.id2word = [self.pad, self.eos, self.cls_token] + list(map(str,np.arange(self.num_of_symbols))) + ['(',')','+','-']

    self.word2id = {}
    for i in np.arange(len(self.id2word)):
      self.word2id[self.id2word[i]] = i

  @property
  def eos_id(self):
    return self.word2id[self.eos]

  @property
  def num_of_symbols(self):
      return 202 #0-100

  @property 
  def max_depth(self):
    return 40
    
  @property
  def train_length(self):
    return 40

  @property
  def dev_length(self):
    return 40
    
  @property
  def forbidden_lengths(self):
      return [5,10,15,20,29,30,31]
      
      
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
        possible_lengths = list(set(np.arange(1,max_length+1)) - set(self.forbidden_lengths))
        length_index = np.random.randint(len(possible_lengths))
        length = possible_lengths[length_index]
        exp_str, _ = binary_math_tree_generator(length, np.arange(1,int(self.num_of_symbols/10)), ['-','+'], max_value, 0, self.max_depth)
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


class ArithmaticSimpleSameLength10(ArithmaticSimple):
  def __init__(self, data_path):
    self.data_path = data_path
    self.task_name = 'arithmatic_simple'
    self.vocab_path = os.path.join(self.data_path,'vocab')

    self.eos = '<eos>'
    self.pad = '<pad>'
    self.cls_token = '<cls>'

    self.load_vocab()
    self.pretrained = False

  @property 
  def max_depth(self):
    return 20
    
  def load_vocab(self):
    self.id2word = [self.pad, self.eos, self.cls_token] + list(map(str,np.arange(self.num_of_symbols))) + ['(',')','+','-']

    self.word2id = {}
    for i in np.arange(len(self.id2word)):
      self.word2id[self.id2word[i]] = i

  @property
  def eos_id(self):
    return self.word2id[self.eos]

  @property
  def num_of_symbols(self):
      return 10 #0-100

  @property
  def train_length(self):
    return 20

  @property
  def dev_length(self):
    return 20

class ArithmaticSimpleSameLength10Depth6(ArithmaticSimpleSameLength10):
  def __init__(self, data_path):
    self.data_path = data_path
    self.task_name = 'arithmatic_simple_samelength10_depth6'
    self.vocab_path = os.path.join(self.data_path,'vocab')

    self.eos = '<eos>'
    self.pad = '<pad>'
    self.cls_token = '<cls>'

    self.load_vocab()
    self.pretrained = False

  @property 
  def max_depth(self):
    return 6
    
    
class ArithmaticSimpleSameLength10Depth4(ArithmaticSimpleSameLength10):
  def __init__(self, data_path):
    self.data_path = data_path
    self.task_name = 'arithmatic_simple_samelength10_depth4'
    self.vocab_path = os.path.join(self.data_path,'vocab')

    self.eos = '<eos>'
    self.pad = '<pad>'
    self.cls_token = '<cls>'

    self.load_vocab()
    self.pretrained = False

  @property 
  def max_depth(self):
    return 4


class ArithmaticSimpleSameLength10Depth2(ArithmaticSimpleSameLength10):
  def __init__(self, data_path):
    self.data_path = data_path
    self.task_name = 'arithmatic_simple_samelength10_depth2'
    self.vocab_path = os.path.join(self.data_path,'vocab')

    self.eos = '<eos>'
    self.pad = '<pad>'
    self.cls_token = '<cls>'

    self.load_vocab()
    self.pretrained = False

  @property
  def num_of_symbols(self):
      return 10 #-10-10

  @property
  def max_depth(self):
    return 2


class ArithmaticSimpleSameLength100Depth2(ArithmaticSimpleSameLength10Depth2):
  def __init__(self, data_path):
    self.data_path = data_path
    self.task_name = 'arithmatic_simple_samelength10_depth2'
    self.vocab_path = os.path.join(self.data_path, 'vocab')

    self.eos = '<eos>'
    self.pad = '<pad>'
    self.cls_token = '<cls>'

    self.load_vocab()
    self.pretrained = False

  @property
  def num_of_symbols(self):
    return 101


class ArithmaticSimpleSameLength21Depth2Zipfian(ArithmaticSimpleSameLength10Depth2):
  def __init__(self, data_path):
    self.data_path = data_path
    self.task_name = 'arithmatic_simple_samelength10_depth2_zipfian'
    self.vocab_path = os.path.join(self.data_path,'vocab')

    self.eos = '<eos>'
    self.pad = '<pad>'
    self.cls_token = '<cls>'

    self.load_vocab()
    self.pretrained = False
    
  @property
  def forbidden_lengths(self):
      return []

  @property
  def num_of_symbols(self):
      return 21 #-10-10

  def load_vocab(self):
    self.id2word = [self.pad, self.eos, self.cls_token] + list(map(str,np.arange(int(-self.num_of_symbols/2),int(self.num_of_symbols/2+1)))) + ['(',')','+','-']

    self.word2id = {}
    for i in np.arange(len(self.id2word)):
      self.word2id[self.id2word[i]] = i


  def generator(self, number_of_examples, mode="train"):
    max_length = self.train_length if mode == "train" else self.dev_length
    possible_lengths = list(set(np.arange(1,max_length+1)) - set(self.forbidden_lengths))

    N = len(possible_lengths)
    numbers = np.arange(1, N + 1, 1)
    weights = np.arange(N + 1, 1, -1)
    a = 0.6
    weights = weights ** (-a)
    weights /= weights.sum()
    bounded_zipf = stats.rv_discrete(name='bounded_zipf', values=(numbers, weights))

    budgets = {}
    max_value = self.num_of_symbols - 1
    max_output_freq = (number_of_examples / self.num_of_symbols) * 2
    print("max_output_freq: ", max_output_freq)
    for i in tqdm(np.arange(number_of_examples)):
      exp = -self.num_of_symbols
      exp_str = '-' + str(self.num_of_symbols)
      while exp < -int(self.num_of_symbols / 2) or exp > int(self.num_of_symbols / 2):

        if mode == "train":
          length_index = bounded_zipf.rvs(size=1)[0] - 1
        else:
          length_index = np.random.randint(len(possible_lengths))

        length = possible_lengths[length_index]
        exp_str, _ = binary_math_tree_generator(length,
                                                np.arange(-int(self.num_of_symbols / 2), int(self.num_of_symbols / 2) + 1),
                                                ['-', '+'], max_value, 0, self.max_depth)
        exp = eval(exp_str)
        if exp >= -int(self.num_of_symbols / 2) and exp <= int(self.num_of_symbols / 2):
          if exp not in budgets:
            budgets[exp] = 1
          budgets[exp] += 1
          if budgets[exp] > max_output_freq:
            exp = -self.num_of_symbols
            exp_str = '-' + str(self.num_of_symbols)


      exp_tokens = exp_str.split() + [self.eos]
      output = [str(exp)]
      example = {'inputs': self.encode(exp_tokens),
                 'targets':self.encode(output),
                 'inputs_length': len(exp_tokens),
                 'targets_length': len(output)}

      yield example


class ArithmaticSimpleSameLength201Depth2Zipfian(ArithmaticSimpleSameLength21Depth2Zipfian):
  def __init__(self, data_path):
    self.data_path = data_path
    self.task_name = 'arithmatic_simple_samelength10_depth2_zipfian'
    self.vocab_path = os.path.join(self.data_path, 'vocab')

    self.eos = '<eos>'
    self.pad = '<pad>'
    self.cls_token = '<cls>'

    self.load_vocab()
    self.pretrained = False

  @property
  def train_length(self):
    return 40

  @property
  def dev_length(self):
    return 40

  @property
  def forbidden_lengths(self):
    return []

  @property
  def num_of_symbols(self):
    return 201


class ArithmaticSimpleSameLength21Depth2Normal(ArithmaticSimpleSameLength10Depth2):
  def __init__(self, data_path):
    self.data_path = data_path
    self.task_name = 'arithmatic_simple_samelength21_depth2_normal'
    self.vocab_path = os.path.join(self.data_path, 'vocab')

    self.eos = '<eos>'
    self.pad = '<pad>'
    self.cls_token = '<cls>'

    self.load_vocab()
    self.pretrained = False

  @property
  def forbidden_lengths(self):
    return []

  @property
  def num_of_symbols(self):
    return 21  # -10-10

  def load_vocab(self):
    self.id2word = [self.pad, self.eos, self.cls_token] + list(
      map(str, np.arange(int(-self.num_of_symbols / 2), int(self.num_of_symbols / 2 + 1)))) + ['(', ')', '+', '-']

    self.word2id = {}
    for i in np.arange(len(self.id2word)):
      self.word2id[self.id2word[i]] = i

  def generator(self, number_of_examples, mode="train"):
    max_length = self.train_length if mode == "train" else self.dev_length
    possible_lengths = list(set(np.arange(1, max_length + 1)) - set(self.forbidden_lengths))

    N = len(possible_lengths)


    budgets = {}
    max_value = self.num_of_symbols - 1
    max_output_freq = (number_of_examples / self.num_of_symbols) * 2
    print("max_output_freq: ", max_output_freq)
    for i in tqdm(np.arange(number_of_examples)):
      exp = -self.num_of_symbols
      exp_str = '-' + str(self.num_of_symbols)
      while exp < -int(self.num_of_symbols / 2) or exp > int(self.num_of_symbols / 2):

        if mode == "train":
          randomNums = np.maximum(0, np.minimum(np.random.normal(loc=int(N/2), scale=int(N/5), size=1), N-1))
          length_index = int(np.round(randomNums)[0])
        else:
          length_index = np.random.randint(len(possible_lengths))

        length = possible_lengths[length_index]
        exp_str, _ = binary_math_tree_generator(length,
                                                np.arange(-int(self.num_of_symbols / 2),
                                                          int(self.num_of_symbols / 2) + 1),
                                                ['-', '+'], max_value, 0, self.max_depth)
        exp = eval(exp_str)
        if exp >= -int(self.num_of_symbols / 2) and exp <= int(self.num_of_symbols / 2):
          if exp not in budgets:
            budgets[exp] = 1
          budgets[exp] += 1
          if budgets[exp] > max_output_freq:
            exp = -self.num_of_symbols
            exp_str = '-' + str(self.num_of_symbols)

      exp_tokens = exp_str.split() + [self.eos]
      output = [str(exp)]
      example = {'inputs': self.encode(exp_tokens),
                 'targets': self.encode(output),
                 'inputs_length': len(exp_tokens),
                 'targets_length': len(output)}

      yield example

class ArithmaticSimpleSameLength21Depth2NormalBiLing(ArithmaticSimpleSameLength21Depth2Normal):
  def __init__(self, data_path):
    self.data_path = data_path
    self.task_name = 'arithmatic_simple_samelength21_depth2_normal_biling'
    self.vocab_path = os.path.join(self.data_path, 'vocab')

    self.eos = '<eos>'
    self.pad = '<pad>'
    self.cls_token = '<cls>'

    self.load_vocab()
    self.pretrained = False

  @property
  def forbidden_lengths(self):
    return []

  @property
  def num_of_symbols(self):
    return 21  # -10-10

  def load_vocab(self):
    self.id2word = [self.pad, self.eos, self.cls_token]

    symbols = list(map(str, np.arange(int(-self.num_of_symbols / 2),
                                      int(self.num_of_symbols / 2 + 1)))) + [ '+', '-']
    translated_symbols = eq2str(symbols)
    shared_symbols = ['(', ')']
    self.id2word = self.id2word + symbols + translated_symbols + shared_symbols

    self.word2id = {}
    for i in np.arange(len(self.id2word)):
      self.word2id[self.id2word[i]] = i

  def generator(self, number_of_examples, mode="train"):
    max_length = self.train_length if mode == "train" else self.dev_length
    possible_lengths = list(set(np.arange(1, max_length + 1)) - set(self.forbidden_lengths))

    N = len(possible_lengths)


    budgets = {}
    max_value = self.num_of_symbols - 1
    max_output_freq = (number_of_examples / self.num_of_symbols) * 2
    print("max_output_freq: ", max_output_freq)
    for i in tqdm(np.arange(number_of_examples)):
      exp = -self.num_of_symbols
      exp_str = '-' + str(self.num_of_symbols)
      while exp < -int(self.num_of_symbols / 2) or exp > int(self.num_of_symbols / 2):

        if mode == "train":
          randomNums = np.maximum(0, np.minimum(np.random.normal(loc=int(N/2), scale=int(N/5), size=1), N-1))
          length_index = int(np.round(randomNums)[0])
        else:
          length_index = np.random.randint(len(possible_lengths))

        length = possible_lengths[length_index]
        exp_str, _ = binary_math_tree_generator(length,
                                                np.arange(-int(self.num_of_symbols / 2),
                                                          int(self.num_of_symbols / 2) + 1),
                                                ['-', '+'], max_value, 0, self.max_depth)
        exp = eval(exp_str)
        if exp >= -int(self.num_of_symbols / 2) and exp <= int(self.num_of_symbols / 2):
          if exp not in budgets:
            budgets[exp] = 1
          budgets[exp] += 1
          if budgets[exp] > max_output_freq:
            exp = -self.num_of_symbols
            exp_str = '-' + str(self.num_of_symbols)

      if np.random.choice([0,1],1,[0.5,0.5])[0] > 0:
        exp_tokens = eq2str(exp_str.split()) + [self.eos]
      else:
        exp_tokens = exp_str.split() + [self.eos]

      if np.random.choice([0, 1], 1, [0.5, 0.5])[0] > 0:
        output = [str(exp)]
      else:
        output = eq2str([str(exp)])

      example = {'inputs': self.encode(exp_tokens),
                 'targets': self.encode(output),
                 'inputs_length': len(exp_tokens),
                 'targets_length': len(output)}

      yield example


class ArithmaticSimpleSameLength201Depth2Normal(ArithmaticSimpleSameLength21Depth2Normal):
  def __init__(self, data_path):
    self.data_path = data_path
    self.task_name = 'arithmatic_simple_samelength201_depth2_normal'
    self.vocab_path = os.path.join(self.data_path, 'vocab')

    self.eos = '<eos>'
    self.pad = '<pad>'
    self.cls_token = '<cls>'

    self.load_vocab()
    self.pretrained = False

  @property
  def forbidden_lengths(self):
    return []

  @property
  def num_of_symbols(self):
    return 201  # -10-10






if __name__ == '__main__':
#  bin_iden = ArithmaticSimple('data/arithmatic_simple')
#
#  bin_iden.build_tfrecords(10000, 'train')
#  bin_iden.build_tfrecords(2000, 'dev')
#  bin_iden.build_tfrecords(2000, 'test')
#
#  bin_iden = ArithmaticSameLength('data/arithmatic_samelength')
#
#  bin_iden.build_tfrecords(10000, 'train')
#  bin_iden.build_tfrecords(2000, 'dev')
#  bin_iden.build_tfrecords(2000, 'test')
#  
#  
#  bin_iden = ArithmaticSimpleSameLength('data/arithmatic_simple_samelength')
#
#  bin_iden.build_tfrecords(10000, 'train')
#  bin_iden.build_tfrecords(2000, 'dev')
#  bin_iden.build_tfrecords(2000, 'test')
#  
#  bin_iden = ArithmaticSimpleCurriculumLength('data/arithmatic_simple_curriculum_length')
#
#  bin_iden.build_tfrecords(10000, 'train')
#  bin_iden.build_tfrecords(2000, 'dev')
#  bin_iden.build_tfrecords(2000, 'test')

#  bin_iden = ArithmaticSimpleSameLength10Depth2('data/arithmatic_simple_samelength10_depth2')
#
#  bin_iden.build_tfrecords(10000, 'train')
#  bin_iden.build_tfrecords(2000, 'dev')
#  bin_iden.build_tfrecords(2000, 'test')
#
#  bin_iden = ArithmaticSimpleSameLength10Depth4('data/arithmatic_simple_samelength10_depth4')
#
#  bin_iden.build_tfrecords(10000, 'train')
#  bin_iden.build_tfrecords(2000, 'dev')
#  bin_iden.build_tfrecords(2000, 'test')
#  
#  bin_iden = ArithmaticSimpleSameLength10Depth6('data/arithmatic_simple_samelength10_depth6')
#
#  bin_iden.build_tfrecords(10000, 'train')
#  bin_iden.build_tfrecords(2000, 'dev')
#  bin_iden.build_tfrecords(2000, 'test')

  bin_iden = ArithmaticSimpleSameLength21Depth2NormalBiLing('data/arithmatic_simple_samelength21_depth2_normal_biling')

  bin_iden.build_tfrecords(100000, 'train')
  bin_iden.build_tfrecords(2000, 'dev')
  bin_iden.build_tfrecords(2000, 'test')

