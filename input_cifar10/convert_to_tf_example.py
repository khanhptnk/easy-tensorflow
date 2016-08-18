import tensorflow as tf

import os
import logging
import sys
sys.path.append("../utils")
from mylogging import logger

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("data_dir", "cifar-10-batches-py", "Directory contains data")

tf.flags.DEFINE_string("output_dir", "../data_cifar10", "Directory to store outputs")

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
      logger.info("Writing batch " + str(i) + "/" + str(num_examples))
      image = [int(x) for x in images[i, :]]
      label = labels[i]
      example = tf.train.Example()
      features_map = example.features.feature
      features_map["image"].int64_list.value.extend(list(image))
      features_map["label"].int64_list.value.append(label)
      writer.write(example.SerializeToString())

def main(_):
  for i in xrange(NUM_FILE_BATCHES):
    data = unpickle(os.path.join(FLAGS.data_dir, "data_batch_" + str(i + 1)))
    convert(data, os.path.join(FLAGS.output_dir, "data_batch_" + str(i + 1)))

  data = unpickle(os.path.join(FLAGS.data_dir,"test_batch"))
  convert(data, os.path.join(FLAGS.output_dir, "test_batch"))

if __name__ == "__main__":
  tf.app.run()

