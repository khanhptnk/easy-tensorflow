"""Specify Cifar10's convoluational network model"""

import tensorflow as tf
import tensorflow.contrib.slim as slim

import sys
sys.path.insert(0, "../utils")

from hparams import HParams
from input_reader_cifar10 import InputReaderCifar10
from model_base import ModelBase

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("model_hparams",
                       "nums_conv_filters=[64, 64],"
                       "conv_filter_sizes=[5, 5],"
                       "pooling_size=2,"
                       "pooling_stride=2,"
                       "dropout_prob=0.5,"
                       "regularize_constant=0.0005,"
                       "init_stddev=0.01",
                       """
                       Model hyperparameters:
                         nums_pool_conv_layers: number of max-pooling
                           convolutional layers.
                         conv_filter_sizes: sizes of filters of each
                           convolutional layers.
                         pooling_size: size of the pooling kernels.
                         pooling_stride: pooling stride.
                         dropout_prob: probability of keeping activations.
                         regularize_constant: L2 regularizing constant.
                         init_stddev: standard deviation of the Gaussian
                           distribution used for initatializing weights.
                       """)

class ModelCifar10(ModelBase):
  def __init__(self, is_training):
    """Initialize a `ModelCifar10` object.
      Args:
        is_training: a bool, whether the model is used for training or
          evaluation.
    """
    self._hparams = HParams(nums_conv_filters=[64, 64],
                            conv_filter_sizes=[3, 3],
                            pooling_size=2,
                            pooling_stride=2,
                            dropout_prob=0.5,
                            regularize_constant=0.004,
                            init_stddev=5e-2)
    if FLAGS.model_hparams:
      self._hparams.parse(FLAGS.model_hparams)

    super(ModelCifar10, self).__init__(is_training)

  def arg_scope(self):
    """Configure the neural network's layers."""
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
                          self._hparams.regularize_constant),
                        activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params) as sc:
      return sc

  def compute(self, inputs):
    """Compute a batch of outputs of the neural network from a batch of inputs.
      Args:
        inputs: a tensorflow tensor, a batch of input images. Each image is of
          size InputReaderCifar10.IMAGE_SIZE x InputReaderCifar10.IMAGE_SIZE x
          InputReaderCifar10.NUM_CHANNELS.
      Returns:
        net: a tensorflow op, output of the network.
        embedding: a tensorflow op, output of the embedding layer (the second
          last fully connected layer).
    """
    hparams = self._hparams
    net = None
    num_pool_conv_layers = len(hparams.nums_conv_filters)
    for i in xrange(num_pool_conv_layers):
      net = slim.conv2d(inputs if i == 0 else net,
                        hparams.nums_conv_filters[i],
                        [hparams.conv_filter_sizes[i], hparams.conv_filter_sizes[i]],
                        padding="SAME",
                        biases_initializer=tf.constant_initializer(0.1 * i),
                        scope="conv_{0}".format(i))
      net = slim.max_pool2d(net,
                            [hparams.pooling_size, hparams.pooling_size],
                            hparams.pooling_stride,
                            scope="pool_{0}".format(i))

    net = slim.flatten(net, scope="flatten")
    net = slim.fully_connected(net,
                               384,
                               biases_initializer=tf.constant_initializer(0.1),
                               scope="fc_{0}".format(num_pool_conv_layers))

    net = slim.dropout(net,
                       hparams.dropout_prob,
                       scope="dropout_{0}".format(num_pool_conv_layers))

    embedding = slim.fully_connected(net,
                                     192,
                                     biases_initializer=tf.constant_initializer(0.1),
                                     scope="fc_{0}".format(num_pool_conv_layers + 1))

    net = slim.fully_connected(embedding,
                               InputReaderCifar10.NUM_CLASSES,
                               activation_fn=None,
                               biases_initializer=tf.constant_initializer(0.0),
                               scope="fc_{0}".format(num_pool_conv_layers + 2))

    return net, embedding



