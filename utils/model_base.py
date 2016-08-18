
class ModelBase(object):

  def __init__(self, is_training):
    self.is_training = is_training

  def arg_scope(self):
    raise NotImplementedError

  def compute(self):
    raise NotImplementedError
