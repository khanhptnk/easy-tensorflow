import tensorflow as tf

import sys
sys.path.insert(0, "../utils")

from model_base import ModelBase

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("model_hparams",
                       "num_hidden_layers=4,"
                       "conv_filter_sizes=[64, 128, 256, 128],"
                       "pooling_size=2,"
                       "pooling_stride=2,"
                       "embedding_size=256,"
                       "dropout_prob=0.5,"
                       "regularize_constant=0.0005,"
                       "init_stddev=0.01",
                       """
                       Model hyperparameters:
                         num_hidden_layers=
                       """)

class Model(ModelBase):

  def __init__(self, is_training):
    self._hparams = HParams(num_hidden_layers=4,
                            conv_filter_sizes=[64, 128, 256, 128],
                            pooling_size=2,
                            pooling_stride=2,
                            embedding_size=256,
                            dropout_prob=0.5,
                            regularize_constant=0.0005,
                            init_stddev=0.01)
    if FLAGS.model_hparams:
      self._hparams = self._hparams.parse(FLAGS.model_hparams)

    super(Model, self).__init__(is_training)

  def arg_scope(self):
    batch_norm_params = {
      "is_training" : self.is_training,
      "decay" : 0.9997,
      "epsilon" : 0.001,
      "variables_collections" : {
        "beta" : None,
        "gamma" : None,
        "moving_mean" : ["moving_vars"],
        "moving_variance" : ["moving_vars"]
      }
    }

    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_initializer=tf.truncated_normal_initializer(
                          stddev=self._hparams.init_stddev),
                        weights_regularizer=slim.l2_regularizer(
                          self._hparams.regularize_constant)
                        activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params) as sc:
      return sc

  def compute(self, inputs):
    hparams = self._hparams
    net = None
    for i in xrange(hparams.num_hidden_layers):
      net = slim.conv2d(inputs if i == 0 else net,
                        hparams.num_conv_filters[i],
                        [hparams.conv_filter_sizes[i], hparams.conv_filter_sizes[i]],
                        padding="SAME",
                        scope="conv_{0}".format(i))
      net = slim.max_pool2d(net,
                            [hparams.pooling_size, hparams.pooling_size],
                            hparams.pooling_stride,
                            scope="pool_{0}".format(i))
    net = slim.flatten(net, scope="flatten")
    net = slim.fully_connected(net, 4096, scope="fc_{0}".format(hparams.num_hidden_layers))

    net = slim.dropout(net,
                       hparams.dropout_prob,
                       scope="dropout_{0}".format(hparams.num_hidden_layers))

    embedding = slim.fully_connected(net,
                                     hparams.embedding_size,
                                     activation_fn=None,
                                     scope="fc_{0}".format(hparams.num_hidden_layers + 1))

    net = slim.fully_connected(embedding,
                               problem_config.NUM_CLASSES,
                               activation_fn=None,
                               scope="fc_{0}".format(hparams.num_hidden_layers + 2))

    return net, embedding



