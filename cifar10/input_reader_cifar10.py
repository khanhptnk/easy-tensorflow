"""Read Cifar10 labeled images and make batches of data."""

import tensorflow as tf

import sys
sys.path.insert(0, "../utils")

from input_reader_base import InputReaderBase
from tensorflow.python.platform import gfile

FLAGS = tf.flags.FLAGS

class InputReaderCifar10(InputReaderBase):
  NUM_CLASSES = 10
  NUM_CHANNELS = 3
  IMAGE_SIZE = 32
  IMAGE_CROPPED_SIZE = 24

  def read_input(self, data_path, batch_size, randomize_input=True,
                 distort_inputs=True, name="read_input"):
    """Read input labeled images and make a batch of examples.

      Labeled images are read from files of tf.Example protos. This proto has
      to contain two features: `image` and `label`, corresponding to an image
      and its label. After being read, the labeled images are put into queues
      to make a batch of examples every time the batching op is executed.

      Args:
        data_path: a string, path to files of tf.Example protos containing
          labeled images.
        batch_size: a int, number of labeled images in a batch.
        randomize_input: a bool, whether the images in the batch are randomized.
        distort_inputs: a bool, whether to distort the images.
        name: a string, name of the op.
      Returns:
        keys: a tensowflow op, the keys of tf.Example protos.
        examples: a tensorflow op, a batch of examples containing labeled
          images. After being materialized, this op becomes a dict, in which the
          `decoded_observation` key is an image and the `decoded_label` is the
          label of that image.
    """
    feature_types = {}
    feature_types["image"] = tf.FixedLenFeature(
        shape=[3072,], dtype=tf.int64, default_value=None)

    feature_types["label"] = tf.FixedLenFeature(
        shape=[1,], dtype=tf.int64, default_value=None)

    keys, examples = tf.contrib.learn.io.graph_io.read_keyed_batch_examples(
        file_pattern=data_path,
        batch_size=batch_size,
        reader=tf.TFRecordReader,
        randomize_input=randomize_input,
        queue_capacity=batch_size * 4,
        num_threads=10 if randomize_input else 1,
        parse_fn=lambda example_proto: self._preprocess_input(example_proto,
                                                              feature_types,
                                                              distort_inputs),
        name=name)

    return keys, examples

  def _preprocess_input(self, example_proto, feature_types, distort_inputs):
    """Parse an tf.Example proto and preprocess its image and label.

      Args:
        example_proto: a tensorflow op, a tf.Example proto.
        feature_types: a dict, used for parsing a tf.Example proto. This is the
          same `feature_types` dict constructed in the `read_input` method.
        distort_inputs: a bool, whether to distort the images.
      Returns:
        example: a tensorflow op, after being materialized becomes a dict, in
          in which the `decoded_observation` key is a processed image, a tensor
          of size InputReaderCifar10.IMAGE_SIZE x
          InputReaderCifar10.IMAGE_SIZE x InputReaderCifar10.NUM_CHANNELS and
          the `decoded_label` is the label of that image, a vector of size
          InputReaderCifar10.NUM_CLASSES.
    """
    example = tf.parse_single_example(example_proto, feature_types)
    image = tf.reshape(example["image"], [InputReaderCifar10.NUM_CHANNELS,
                                          InputReaderCifar10.IMAGE_SIZE,
                                          InputReaderCifar10.IMAGE_SIZE])
    image = tf.transpose(image, perm=[1, 2, 0])
    image = tf.cast(image, tf.float32)
    if distort_inputs:
      image = tf.random_crop(image, [InputReaderCifar10.IMAGE_CROPPED_SIZE,
                                     InputReaderCifar10.IMAGE_CROPPED_SIZE,
                                     3])
      image = tf.image.random_flip_left_right(image)
      image = tf.image.random_brightness(image, max_delta=63)
      image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
    else:
      image = tf.image.resize_image_with_crop_or_pad(image,
          InputReaderCifar10.IMAGE_CROPPED_SIZE,
          InputReaderCifar10.IMAGE_CROPPED_SIZE)
    image = tf.image.per_image_whitening(image)
    example["decoded_observation"] = image

    label = tf.one_hot(example["label"], InputReaderCifar10.NUM_CLASSES, on_value=1, off_value=0)
    label = tf.reshape(label, [InputReaderCifar10.NUM_CLASSES,])
    label = tf.cast(label, tf.int64)
    example["decoded_label"] = label

    return example

