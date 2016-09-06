"""Convert raw Cifar10 data to tf.Example protos.

  Assumed that you had downloaded the CIFAR-10 python version from:
    http://www.cs.toronto.edu/~kriz/cifar.html
  and extracted it to `FLAGS.data_dir` directory.
"""

import tensorflow as tf

import os
import sys
sys.path.append("../utils")

from tensorflow.python.platform import tf_logging as logging

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("data_dir", "cifar-10-batches-py", "Directory contains data.")

tf.flags.DEFINE_string("output_dir", "../tmp/", "Directory to store outputs.")

NUM_FILE_BATCHES = 5

def unpickle(file):
  import cPickle
  fo = open(file, 'rb')
  dict = cPickle.load(fo)
  fo.close()
  return dict

def convert(data, filename):
  images = data["data"]
  labels = data["labels"]
  num_examples = images.shape[0]
  with tf.python_io.TFRecordWriter(filename) as writer:
    for i in xrange(num_examples):
      logging.info("Writing batch " + str(i) + "/" + str(num_examples))
      image = [int(x) for x in images[i, :]]
      label = labels[i]
      example = tf.train.Example()
      features_map = example.features.feature
      features_map["image"].int64_list.value.extend(list(image))
      features_map["label"].int64_list.value.append(label)
      writer.write(example.SerializeToString())

def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  for i in xrange(NUM_FILE_BATCHES):
    data = unpickle(os.path.join(FLAGS.data_dir, "data_batch_" + str(i + 1)))
    convert(data, os.path.join(FLAGS.output_dir, "data_batch_" + str(i + 1)))

  data = unpickle(os.path.join(FLAGS.data_dir,"test_batch"))
  convert(data, os.path.join(FLAGS.output_dir, "test_batch"))

if __name__ == "__main__":
  tf.app.run()

