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
    """Configure model's layers.

      Raises:
        NotImplementedError: this method is required to be implemented by any
          subclass of `ModelBase`.
    """
    raise NotImplementedError

  def compute(self):
    """Compute a batch of outputs of the model from a batch of inputs.

      Raises:
        NotImplementedError: this method is required to be implemented by any
          subclass of `ModelBase`.
    """
    raise NotImplementedError
