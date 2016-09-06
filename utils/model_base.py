"""Define a class to specify a tensorflow model architecture."""

class ModelBase(object):
  def __init__(self, is_training):
    """Initialize a `ModelBase` object.
      Args:
        is_training: a bool, whether the model is used for training or
          evaluation.
    """
    self.is_training = is_training

  def arg_scope(self):
    """Configurate model's layers.

      Raises:
        NotImplementedError: this method is required to be implemented by any
          subclass of `ModelBase`.
    """
    raise NotImplementedError

  def compute(self):
    """Compute output of the model from a batch of examples.

      Raises:
        NotImplementedError: this method is required to be implemented by any
          subclass of `ModelBase`.
    """
    raise NotImplementedError
