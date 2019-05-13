import tensorflow as tf
import collections
import os
from tqdm import tqdm
import numpy as np

class PCFG(object):

  def __init__(self, data_path):
    self.data_path = data_path
    self.eos = '<eos>'
    self.pad = '<pad>'
    self.unk = '<unk>'
    self.pretrained = False

    self.load_vocab()

  @property
  def share_input_output_embeddings(self):
    return False

  @property
  def vocab_length(self):
    return len(self.input_id2word)

  @property
  def target_vocab(self):
    return self.target_id2word

  @property
  def target_length(self):
    return None

  @property
  def eos_id(self):
    return self.target_word2id[self.eos]

  def read_sentences(self, filename):
    input_sentences = []
    target_sentences = []
    with tf.gfile.GFile(filename+'.src', "r") as f:
      lines = f.read().split("\n")
      for line in lines:
        input_sentence = line
        input_sentences.append(input_sentence.split() + [self.eos])

        with tf.gfile.GFile(filename+'.tgt', "r") as f:
          lines = f.read().split("\n")
          for line in lines:
            target_sentence = line
            target_sentences.append(target_sentence.split() + [self.eos])

    return input_sentences, target_sentences

  def read_words(self, filename):
    input_sentences, target_sentences = self.read_sentences(filename)
    return [word for sent in input_sentences for word in sent], \
           [word for sent in target_sentences for word in sent]


  def build_input_and_target_vocabs(self, filename):
    input_words, target_words = self.read_words(filename)
    self.build_vocab(input_words, prefix="input", add_unknown=True)
    self.build_vocab(target_words, prefix="target")

  def build_vocab(self, data, prefix, add_unknown=False):
    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))


    if add_unknown:
      if self.unk not in word_to_id:
        word_to_id[self.unk] = len(word_to_id)

    id_to_word = {}
    for word,id in word_to_id.items():
      id_to_word[id] = word

    print("vocab size: ", len(word_to_id))
    vocab_dict = {'word_to_id': word_to_id, 'id_to_word': id_to_word}

    np.save(os.path.join(self.data_path, prefix+"_vocab"), vocab_dict)

  def load_vocab(self):

    input_vocab_path = os.path.join(self.data_path, 'input_vocab.npy')
    if not os.path.exists(input_vocab_path):
      self.build_input_and_target_vocabs(os.path.join(self.data_path,'train'))

    input_vocab = np.load(input_vocab_path, allow_pickle=True).item()

    self.input_word2id = input_vocab['word_to_id']
    self.input_id2word = input_vocab['id_to_word']

    target_vocab_path = os.path.join(self.data_path, 'target_vocab.npy')
    target_vocab = np.load(target_vocab_path, allow_pickle=True).item()

    self.target_word2id = target_vocab['word_to_id']
    self.target_id2word = target_vocab['id_to_word']

  def encode_target(self, sent):
    encoded_sent = []
    for word in sent:
      if word not in self.target_word2id:
        word = self.unk
      encoded_sent.append(self.target_word2id[word])

    return encoded_sent

  def encode_input(self, sent):
    encoded_sent = []
    for word in sent:
      if word not in self.input_word2id:
        word = self.unk
      encoded_sent.append(self.input_word2id[word])

    return encoded_sent

  @staticmethod
  def decode(encoded_sent, id_to_word):
    sent = []
    for id in encoded_sent:
      sent.append(id_to_word[id])

    return sent

  def example_generator(self, input_sentences, target_sentences):
    for inputs, targets in zip(input_sentences,target_sentences):
      inputs = self.encode_input(inputs)
      targets = self.encode_target(targets)
      example = {}
      example['inputs'] = inputs
      example['targets'] = targets
      example['inputs_length'] = len(inputs)
      example['targets_length'] = len(targets)

      yield example

  def get_tf_example(self,example):
    """Convert our own representation of an example's features to Features class for TensorFlow dataset.
    """
    features = tf.train.Features(feature={
      "inputs": tf.train.Feature(int64_list=tf.train.Int64List(value=example['inputs'])),
      "targets": tf.train.Feature(int64_list=tf.train.Int64List(value=example['targets'])),
      "inputs_length": tf.train.Feature(int64_list=tf.train.Int64List(value=[example['inputs_length']])),
      "targets_length": tf.train.Feature(int64_list=tf.train.Int64List(value=[example['targets_length']]))
    })
    return features

  def build_tfrecords(self, mode):
    data = self.read_sentences(os.path.join(self.data_path, mode))
    tf_examples = []
    for example in self.example_generator(*data):
       tf_examples.append(self.get_tf_example(example))

    with tf.python_io.TFRecordWriter(os.path.join(self.data_path,mode + ".tfr")) as tf_record_writer:
      for example in tqdm(tf_examples):
        tf_record = tf.train.Example(features=example)
        tf_record_writer.write(tf_record.SerializeToString())

  def parse_examples(self, example):
      """Load an example from TF record format."""
      features = {"inputs_length": tf.FixedLenFeature([], tf.int64),
                  "targets_length": tf.FixedLenFeature([], tf.int64),
                  "inputs": tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
                  "targets": tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
                  }
      parsed_example = tf.parse_single_example(example, features=features)

      inputs_length = parsed_example["inputs_length"]
      targets_lengths = parsed_example['targets_length']
      inputs = parsed_example["inputs"]
      targets = parsed_example["targets"]

      return inputs, targets, inputs_length, targets_lengths

  def get_padded_shapes(self,):
    return [None], [None], [], []

  def get_tfrecord_path(self, mode):
    return os.path.join(self.data_path, mode + ".tfr")




if __name__ == '__main__':
  pcfg = PCFG('data/pcfg')
  pcfg.build_tfrecords("train")
  pcfg.build_tfrecords("dev")
  pcfg.build_tfrecords("test")

  batch_size = 10
  dataset = tf.data.TFRecordDataset(pcfg.get_tfrecord_path(mode="train"))
  dataset = dataset.map(pcfg.parse_examples)
  dataset = dataset.padded_batch(batch_size, padded_shapes=pcfg.get_padded_shapes())
  iterator = dataset.make_initializable_iterator()

  example = iterator.get_next()
  inputs, targets, inputs_length, targets_length = example

  global_step = tf.train.get_or_create_global_step()
  scaffold = tf.train.Scaffold(local_init_op=tf.group(tf.local_variables_initializer(),
                                                      iterator.initializer))
  with tf.train.MonitoredTrainingSession(checkpoint_dir='logs/tests', scaffold=scaffold) as sess:
    inp, targ, tag_len = \
      sess.run([inputs, targets, targets_length])
    print(pcfg.decode(inp[0], pcfg.input_id2word))
    print(pcfg.decode(targ[0], pcfg.target_id2word))
    print(tag_len)