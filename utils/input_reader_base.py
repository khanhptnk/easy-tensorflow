"""Define a class for reading input from files of tf.Example protos."""

class InputReaderBase(object):
  def __init__(self):
    """Initialize an `InputReaderBase` object."""
    pass

  def read_input(self, data_path, batch_size, randomize_input=True, name="read_input"):
    """Read input labeled images and make a batch of examples.

      Labeled images are read from files of tf.Example protos. This proto has
      to contain two features: `image` and `label`, corresponding to an image
      and its label. After being read, the labeled images are put into queues
      to make a batch of examples every time the batching op is executed.

      Args:
        data_path: a string, path to files of tf.Example protos containing
          labeled images.
        batch_size: a int, number of labeled images in a batch.
        randomize_input: a bool, whether the images in the batch are randomized.
        name: a string, name of the op.

      Raises:
        NotImplementedError: this method is required to be implemented by any
          subclass of `InputReaderBase`.
    """
    raise NotImplementedError
