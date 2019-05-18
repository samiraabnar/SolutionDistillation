import os
import tarfile

import tensorflow as tf


class SentimentYelpFull(object):
  """Yelp dataset."""
  URL = "https://s3.amazonaws.com/fast-ai-nlp/yelp_review_full_csv.tgz"

  def __init__(self, data_dir):
    self.data_dir = data_dir

  def doc_generator(self, mode):
    file_path = os.path.join(self.data_dir,mode+".csv")
    with tf.gfile.Open(file_path) as yelp_f:
      lines = yelp_f.readlines()
      for line in lines:
        label = line[1]
        doc = line[5:-2].strip()
        yield doc, label

  def generate_samples(self, mode, tmp_dir):
    """Generate examples."""
    # Download and extract
    compressed_filename = os.path.basename(self.URL)
    download_path = data_utils.maybe_download(tmp_dir, compressed_filename,
                                                   self.URL)
    yelp_dir = os.path.join(tmp_dir, "yelp_review_full_csv")
    if not tf.gfile.Exists(yelp_dir):
      with tarfile.open(download_path, "r:gz") as tar:
        tar.extractall(tmp_dir)

    # Generate examples
    for doc, label in self.doc_generator(mode):
      yield {
          "inputs": doc,
          "label": int(label),
      }