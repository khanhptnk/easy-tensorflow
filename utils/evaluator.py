"""Define a class for evaluating a trained tensorflow model.

  This class inherits `TrainEvalBase` (defined in `train_eval_base.py`).
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim

import os

from hparams import HParams
from train_eval_base import TrainEvalBase

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("data_path", "../data_cifar10/test_batch",
    "Path to files of tf.Example protos containing evaluating data.")

tf.flags.DEFINE_string("logdir", "/tmp/cifar10/", "Directory where to write event logs")

tf.flags.DEFINE_string("eval_config", "batch_size=16,"
                                      "num_batches=100,"
                                      "eval_interval_secs=50,",
                        """
                        Evaluation configuration:
                          batch_size: an int, size of a batch of examples.
                          num_batches: an int, number of batches of examples.
                          eval_interval_secs: a int, time interval between each
                            evaluation of the model.
                        """)

class Evaluator(TrainEvalBase):
  def __init__(self, model, loss_fn, graph, input_reader):
    """Initialize an `Evaluator` object.

      Args:
        model: an instance of a subclass of the `ModelBase` class (defined in
          `model_base.py`).
        loss_fn: a tensorflow op, a loss function for training a model. See:
            https://www.tensorflow.org/code/tensorflow/contrib/losses/python/losses/loss_ops.py
          for a list of available loss functions.
        graph: a tensorflow computation graph.
        input_reader: an instance of a subclass of the `InputReaderBase` class
          (defined in `input_reader_base.py`).
    """
    self._config = HParams(batch_size=128,
                           num_batches=400,
                           eval_interval_secs=100)

    if FLAGS.eval_config:
      self._config.parse(FLAGS.eval_config)

    super(Evaluator, self).__init__(model, loss_fn, FLAGS.data_path, FLAGS.logdir, graph, input_reader)

  # TODO: make the evaluating metrics customizable.
  def _compute_loss_and_other_metrics(self):
    """Compute loss function and other evaluating metrics."""
    self._compute_loss()

    probabilities = tf.sigmoid(self._outputs)
    predictions = tf.argmax(self._outputs, dimension=1)
    truths = tf.argmax(self._labels, dimension=1)
    metrics_to_values, self._metrics_to_updates = slim.metrics.aggregate_metric_map(
        {
            "Accuracy" : slim.metrics.streaming_accuracy(predictions, truths),
            "Loss_Eval" : slim.metrics.streaming_mean(self._loss),
        })
    for metric_name, metric_value in metrics_to_values.iteritems():
      self._summary_ops.append(tf.scalar_summary(metric_name, metric_value))

  def run(self):
    """Run evaluation."""
    # Create logging directory if not exists.
    if not os.path.isdir(self._eval_log_dir):
      os.makedirs(self._eval_log_dir)

    # Compute loss function and other evaluating metrics.
    self._initialize()

    # Visualize input images in Tensorboard.
    self._summary_ops.append(tf.image_summary("Eval_Image", self._observations, max_images=5))

    # Use `slim.evaluation.evaluation_loop` to evaluate the model periodically.
    slim.evaluation.evaluation_loop(
        master='',
        checkpoint_dir=self._train_log_dir,
        logdir=self._eval_log_dir,
        num_evals=self._config.num_batches,
        eval_op=self._metrics_to_updates.values(),
        summary_op=tf.merge_summary(self._summary_ops),
        eval_interval_secs=self._config.eval_interval_secs)
