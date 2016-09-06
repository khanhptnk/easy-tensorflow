"""Define a base class for training and evaluating a tensorflow model."""

import tensorflow as tf
import tensorflow.contrib.slim as slim

import os

class TrainEvalBase(object):
  def __init__(self, model, loss_fn, data_path, log_dir, graph, input_reader):
    """Initialize a `TrainEvalBase` object.
      Args:
        model: an instance of a subclass of the `ModelBase` class (defined in
          `model_base.py`).
        loss_fn: a tensorflow op, a loss function for training a model. See:
            https://www.tensorflow.org/code/tensorflow/contrib/losses/python/losses/loss_ops.py
          for a list of available loss functions.
        data_path: a string, path to files of tf.Example protos containing data.
        log_dir: a string, logging directory.
        graph: a tensorflow computation graph.
        input_reader: an instance of a subclass of the `InputReaderBase` class
          (defined in `input_reader_base.py`).
    """
    self._data_path = data_path
    self._log_dir = log_dir
    self._train_log_dir = os.path.join(self._log_dir, "train")
    self._eval_log_dir = os.path.join(self._log_dir, "eval")

    self._model = model
    self._loss_fn = loss_fn
    self._graph = graph
    self._input_reader = input_reader

    self._summary_ops = []

  def _load_data(self):
    """Load data from files of tf.Example protos."""
    keys, examples = self._input_reader.read_input(
        self._data_path,
        self._config.batch_size,
        randomize_input=self._model.is_training,
        distort_inputs=self._model.is_training)

    self._observations = examples["decoded_observation"]
    self._labels = examples["decoded_label"]

  def _compute_loss(self):
    """Compute the loss function from outputs of the model and ground truths.

      Other endpoints of the model (such as outputs of other layers) could be
        returned to `self._endpoints`.
    """
    with slim.arg_scope(self._model.arg_scope()):
      self._outputs, self._endpoints = self._model.compute(self._observations)

    self._loss = self._loss_fn(self._outputs, self._labels)

  def _compute_loss_and_other_metrics(self):
    """Compute the loss function and other metrics for evaluation.

      Raises:
        NotImplementedError: this function is required to be implemented by any
          subclass of `TrainEvalBase`.
    """
    raise NotImplementedError

  def _initialize(self):
    """Load data, compute loss function and other evaluating metric."""
    self._load_data()
    self._compute_loss_and_other_metrics()

  def run(self):
    """Run training or evaluating.

      Raises:
        NotImplementedError: this function is required to be implemented by any
          subclass of `TrainEvalBase`.
    """
    raise NotImplementedError

