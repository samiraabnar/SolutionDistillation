import codecs
import os
import tempfile
import zipfile
from six.moves import urllib


data_dir = tempfile.mkdtemp()
print('saving files to %s' % data_dir)

def download_and_unzip(url_base, zip_name, *file_names):
  zip_path = os.path.join(data_dir, zip_name)
  url = url_base + zip_name
  print('downloading %s to %s' % (url, zip_path))
  urllib.request.urlretrieve(url, zip_path)
  out_paths = []
  with zipfile.ZipFile(zip_path, 'r') as f:
    for file_name in file_names:
      print('extracting %s' % file_name)
      out_paths.append(f.extract(file_name, path=data_dir))
  return out_paths


full_glove_path, = download_and_unzip(
  'http://nlp.stanford.edu/data/', 'glove.840B.300d.zip',
  'glove.840B.300d.txt')



train_path, dev_path, test_path = download_and_unzip(
  'http://nlp.stanford.edu/sentiment/', 'trainDevTestTrees_PTB.zip',
  'trees/train.txt', 'trees/dev.txt', 'trees/test.txt')

filtered_glove_path = os.path.join(data_dir, 'filtered_glove.txt')

def filter_glove():
  vocab = set()
  # Download the full set of unlabeled sentences separated by '|'.
  sentence_path, = download_and_unzip(
    'http://nlp.stanford.edu/~socherr/', 'stanfordSentimentTreebank.zip',
    'stanfordSentimentTreebank/SOStr.txt')
  with codecs.open(sentence_path, encoding='utf-8') as f:
    for line in f:
      # Drop the trailing newline and strip backslashes. Split into words.
      vocab.update(line.strip().replace('\\', '').split('|'))
  nread = 0
  nwrote = 0
  with codecs.open(full_glove_path, encoding='utf-8') as f:
    with codecs.open(filtered_glove_path, 'w', encoding='utf-8') as out:
      for line in f:
        nread += 1
        line = line.strip()
        if not line: continue
        if line.split(u' ', 1)[0] in vocab:
          out.write(line + '\n')
          nwrote += 1
  print('read %s lines, wrote %s' % (nread, nwrote))


filter_glove()

