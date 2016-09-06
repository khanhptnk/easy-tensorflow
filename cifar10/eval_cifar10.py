"""Evaluate a trained Cifar10 model."""

import tensorflow as tf

import sys

sys.path.insert(0, "../utils")

from evaluator import Evaluator
from input_reader_cifar10 import InputReaderCifar10
from model_cifar10 import ModelCifar10

FLAGS = tf.flags.FLAGS

# The following flags are defined in `../utils/evaluator.py`
#   data_path: Path to files of tf.Example protos containing evaluating data.
#   logdir: Directory to write event logs and summary stats.
#   eval_config: evaluation configuration.

# The following flag is defined in `model_cifar10.py`
#   model_hparams: hyperparameters of the model.

def main(_):
  # Set logging level.
  tf.logging.set_verbosity(tf.logging.INFO)
  # Create a tensorflow computation graph.
  graph = tf.Graph()
  with graph.as_default():
    with tf.device("/cpu:0"):
      # Create an `Evaluator` object.
      evaluator = Evaluator(model=ModelCifar10(is_training=False),
                      loss_fn=tf.contrib.losses.softmax_cross_entropy,
                      graph=graph,
                      input_reader=InputReaderCifar10())
      # Run evaluation.
      evaluator.run()

if __name__ == "__main__":
  tf.app.run()
