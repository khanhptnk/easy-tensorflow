"""Train a Cifar10 model."""

import tensorflow as tf

import sys

sys.path.insert(0, "../utils")

from input_reader_cifar10 import InputReaderCifar10
from model_cifar10 import ModelCifar10
from trainer import Trainer

FLAGS = tf.flags.FLAGS

# The following flags are defined in `../utils/trainer.py`
#   data_path: Path to files of tf.Example protos containing training data.
#   logdir: directory to write event logs and save model checkpoints.
#   train_config: training configuration.

# The following flag is defined in `model_cifar10.py`
#   model_hparams: hyperparameters of the model.

def main(_):
  # Set logging level.
  tf.logging.set_verbosity(tf.logging.INFO)
  # Create a tensorflow computation graph for training.
  graph = tf.Graph()
  with graph.as_default():
    # Create a `Trainer` object.
    trainer = Trainer(model=ModelCifar10(is_training=True),
                      loss_fn=tf.contrib.losses.softmax_cross_entropy,
                      graph=graph,
                      input_reader=InputReaderCifar10())
    # Run training.
    trainer.run()

if __name__ == "__main__":
  tf.app.run()
