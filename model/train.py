import tensorflow as tf

import sys

sys.path.insert(0, "../utils")

from trainer import Trainer
from model import Model

FLAGS = tf.flags.FLAGS

# The following flags are defined in:
#   ../utils/trainer.py
# DECLARE_string(data_path)
# DECLARE_string(logdir)
# DECLARE_string(master)

# The following flag is defined in:
#   model.py
# DECLARE_string(model_hparams)

def main(_):
  graph = tf.Graph()
  with graph.as_default():
    trainer = Trainer(model=Model(is_training=True),
                      loss_fn=tf.contrib.losses.softmax_cross_entropy,
                      graph=graph)
    trainer.run()

if __name__ == "__main__":
  tf.app.run()
