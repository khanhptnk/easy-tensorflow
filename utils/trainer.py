import tensorflow as tf

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("data_path", "",
    "Path to an SSTable of tf.Example protos training data")

tf.flags.DEFINE_string("logdir", "", "Directory where to write event logs")

tf.flags.DEFINE_string("master", "local", "Name of the TensorFlow master to use")

tf.flags.DEFINE_string("train_config", "learning_rate=0.1,"
                                       "batch_size=128,"
                                       "train_steps=10000,"
                                       "save_summary_secs=100,"
                                       "save_interval_secs=100,",
                        """
                        Training configuration:
                          learning_rate: a float, learning rate of the optimizer.
                          batch_size: an integer, size of a batch of examples.
                          train_steps: an int, number of training steps.
                          save_summary_secs: an int, time interval between each
                            save of summary ops.
                          save_interval_secs: a int, time interval between each
                            save of a model checkpoint.
                        """)

class Trainer(TrainEvalBase):

  def __init__(self, model, loss_fn, graph):
    self._config = HParams(tf_master=FLAGS.master,
                           learning_rate=0.1,
                           batch_size=128,
                           train_steps=10000,
                           save_summary_secs=100,
                           save_interval_secs=100)

    if FLAGS.train_config:
      self._config = self._config.parse(FLAGS.train_config)

    super(Trainer, self).__init__(model, loss_fn, FLAGS.data_path, FLAGS.logdir, graph)

  def _compute_loss_ans_metrics(self):
    self._compute_loss()
    self._summary_ops.append(tf.scalar_summary('Loss_Train', self._loss))

  def run(self):
    if os.path.isdir(self._train_log_dir):
      os.makedirs(self._train_log_dir)

    self._initialize()

    optimizer = tf.train.AdagradOptimizer(self._config.learning_rate)
    train_op = slim.learning.create_train_op(self._loss, optimizer)

    slim.learning.train(train_op=train_op,
                        logdir=self._train_log_dir,
                        graph=self._graph,
                        master=self._config.tf_master,
                        number_of_steps=self._config.train_steps,
                        summary_op=tf.merge_summary(self._summary_ops),
                        save_summary_secs=tf._config.save_summary_secs,
                        save_interval_secs=tf._config.save_interval_secs)

