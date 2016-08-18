import tensorflow as tf

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("data_path", "",
    "Path to an SSTable of tf.Example protos training data")

tf.flags.DEFINE_string("logdir", "", "Directory where to write event logs")

tf.flags.DEFINE_string("master", "local", "Name of the TensorFlow master to use")

tf.flags.DEFINE_string("eval_config", "batch_size=128,"
                                      "num_batches=400,"
                                      "eval_interval_secs=100,",
                        """
                        Evaluation configuration:
                          batch_size: an int, size of a batch of examples.
                          num_batches: an int, number of batches of examples.
                          eval_interval_secs: a int, time interval between each
                            evaluation of the model.
                        """)

class Evaluator(TrainEvalBase):

  def __init__(self, model, loss_fn, graph):
    self._config = HParams(tf_master=FLAGS.master,
                           batch_size=128,
                           num_batches=400,
                           eval_interval_secs=100)

    if FLAGS.eval_config:
      self._config = self._config.parse(FLAGS.eval_config)

    super(Evaluator, self).__init__(model, loss_fn, FLAGS.data_path, FLAGS.logdir, graph)

  def _compute_loss_ans_metrics(self):
    self._compute_loss()

    probabilities = tf.sigmoid(self._outputs)
    metrics_to_values, self._metrics_to_updates = slim.metrics.aggregate_metric_map(
        {
            "AUC" : slim.metrics.auc(probabilities, self._labels),
            "Loss_Eval" : slim.metrics.mean(self._loss),
        })
    for metric_name, metric_value in metrics_to_values.iteritems():
      self._summary_ops.append(tf.scalar_summary(metric_name, metric_value))

  def run(self):
    if os.path.isdir(self._eval_log_dir):
      os.makedirs(self._eval_log_dir)

    self._initialize()

    slim.evaluation.evaluation_loop(
        master=self._config.tf_master,
        checkpoint_dir=self._train_log_dir,
        logdir=self._eval_log_dir,
        num_evals=self._config.num_batches,
        eval_op=self.metrics_to_updates.values(),
        summary_op=tf.merge_summary(self._summary_ops),
        eval_interval_secs=self._config.eval_interval_secs)
