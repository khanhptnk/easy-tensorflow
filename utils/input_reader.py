import tensorflow as tf

def read_input(data_path, batch_size, feature_names, config, randomize_input):
  feature_types = {}
  feature_types[feature_names.IMAGE_FEATURE] = tf.FixedLenFeature(
      shape=[], dtype=tf.string, default_value="")

  feature_types[feature_names.LABEL_FEATURE] = tf.FixedLenFeature(
      shape=[1,], dtype=tf.string, default_value="")

  keys, examples = tf.contrib.learn.io.graph_io.read_keyed_batch_examples(
      file_pattern=data_path,
      batch_size=batch_size,
      reader=tf.TFRecordReader,
      randomize_input=randomize_input,
      queue_capacity=batch_size * 4,
      num_threads=10 if randomize_input else 1)

