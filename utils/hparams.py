class HParams:
  def __init__(self, **kwargs):
    self._types = {}
    for key, value in kwargs.iteritems():
      setattr(self, key, value)
      self._types[key] = type(value)

  def parse(self, arg_string):
    arg_list = arg_string.split(",")
    for arg in arg_list:
      arg_comma_separated = arg.split("=")
      key = arg_comma_separated[0]
      value = arg_comma_separated[1]
      if self._types[key] is bool:
        self._setBool(key, value)
      elif self._types[key] is int:
        self._setInt(key, value)
      elif self._types[key] is float:
        self._setFloat(key, value)
      elif self._types[key] is list:
        self._setList(key, value)

  def _setBool(self, key, value):
    if value == "False":
      setattr(self, key, False)
    elif value == "True":
      setattr(self, key, True)
    else:
      raise ValueError("Expected True or False but received " + value)

  def _setInt(self, key, value):
    try:
      setattr(self, key, int(value))
    except ValueError:
      print "Expected an integer but received " + value


  def _setFloat(self, key, value):
    try:
      setattr(self, key, value)
    except ValueError:
      print "Expected a float but received " + value

  def _setList(self, key, value):
    if not value.startswith("["):
      raise ValueError("List does not start with `[`:" + value)
    if not value.endswith("]"):
      raise ValueError("List does not end with `]`:" + value)
    value = value[1:-1]
    tmp_list = []
    for num in value.split(","):
      try:
        tmp_list.append(float(num))
      except ValueError:
        print "Expected a float but received " + num
    setattr(self, key, tmp_list)


