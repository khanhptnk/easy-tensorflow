import tensorflow as tf
import tensorflow.contrib.slim as slim

import os

from hparams import HParams
from train_eval_base import TrainEvalBase

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("data_path", "../data_cifar10/data_batch_*",
    "Path to a file of tf.Example protos containing training data.")

tf.flags.DEFINE_string("logdir", "/tmp/cifar10", "Directory where to write event logs")

tf.flags.DEFINE_string("train_config", "learning_rate=0.1,"
                                       "batch_size=256,"
                                       "train_steps=60000,"
                                       "save_summaries_secs=100,"
                                       "save_interval_secs=100,",
                        """
                        Training configuration:
                          learning_rate: a float, learning rate of the optimizer.
                          batch_size: an integer, size of a batch of examples.
                          train_steps: an int, number of training steps.
                          save_summaries_secs: an int, time interval between each
                            save of summary ops.
                          save_interval_secs: a int, time interval between each
                            save of a model checkpoint.
                        """)

class Trainer(TrainEvalBase):

  def __init__(self, model, loss_fn, graph, input_reader):
    self._config = HParams(learning_rate=0.1,
                           batch_size=16,
                           train_steps=10000,
                           save_summaries_secs=100,
                           save_interval_secs=100)

    if FLAGS.train_config:
      self._config.parse(FLAGS.train_config)

    super(Trainer, self).__init__(model, loss_fn, FLAGS.data_path,
                                  FLAGS.logdir, graph, input_reader)

  def _compute_loss_and_other_metrics(self):
    self._compute_loss()
    self._summary_ops.append(tf.scalar_summary('Loss_Train', self._loss))

  def run(self):
    if not os.path.isdir(self._train_log_dir):
      os.makedirs(self._train_log_dir)

    self._initialize()

    self._summary_ops.append(tf.image_summary("Image_Train", self._observations, max_images=5))

    optimizer = tf.train.AdagradOptimizer(self._config.learning_rate)
    train_op = slim.learning.create_train_op(self._loss, optimizer)

    slim.learning.train(train_op=train_op,
                        logdir=self._train_log_dir,
                        graph=self._graph,
                        number_of_steps=self._config.train_steps,
                        summary_op=tf.merge_summary(self._summary_ops),
                        save_summaries_secs=self._config.save_summaries_secs,
                        save_interval_secs=self._config.save_interval_secs)

