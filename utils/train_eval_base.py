import os

import tensorflow as tf

import tensorflow.contrib.slim as slim

class TrainEvalBase(object):
  def __init__(self, model, loss_fn, data_path, log_dir, graph):
    self._data_path = data_path
    self._log_dir = log_dir
    self._train_log_dir = os.path.join(self._log_dir, 'train')
    self._eval_log_dir = os.path.join(self._log_dir, 'eval')

    self._model = model
    self._loss_fn = loss_fn
    self._graph = graph

    self._summary_ops = []

  def _load_data(self):
    keys, examples = input_reader.read_input(
        self._data_path,
        self._config,
        randomize_input=self._model.is_training)

    self._observations = self._config.preprocess_fn(examples["decoded_observation"])
    self._labels = examples["decoded_label"]

  def _compute_loss(self):
    with slim.arg_scope(self._model.arg_scope()):
      self._output, self._endpoints = self._model.compute(self._observations)

    self._loss = self._loss_fn(self._outputs, self._labels)

  def _compute_loss_and_othe_metrics(self):
    raise NotImplementedError

  def _intialize(self):
    self._load_data()
    self._compute_loss_and_metrics()

  def run(self):
    raise NotImplementedError

