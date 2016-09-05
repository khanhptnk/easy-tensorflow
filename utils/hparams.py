"""Define a class for holding model hyperparameters."""

class HParams:
  def __init__(self, **kwargs):
    """Initialize a `HParams` object.

      Args:
        kwargs: a list, each parameter in this list is a model hyperparameter,
          which can be an int, a float, a bool, or a list.
    """
    self._types = {}
    for key, value in kwargs.iteritems():
      setattr(self, key, value)
      self._types[key] = type(value)

  def parse(self, arg_string):
    """Parse and set hyperameters from a comma-separated string."""
    arg_list = self._parse_comma_separated(arg_string)
    for arg in arg_list:
      arg_comma_separated = arg.split("=")
      key = arg_comma_separated[0].strip()
      value = arg_comma_separated[1].strip()
      if self._types[key] is bool:
        self._setBool(key, value)
      elif self._types[key] is int:
        self._setInt(key, value)
      elif self._types[key] is float:
        self._setFloat(key, value)
      elif self._types[key] is list:
        self._setList(key, value)

  def _parse_comma_separated(self, string):
    """Split a comma-separated string.

      The string is not split at commas within a list representation (e.g.
        `[1,2,4]`)

      Returns:
        separated_items: a list, each element is a hyperparameter specification.
    """
    separated_items= []
    lastComma = 0
    openBrackets = 0
    for i in xrange(len(string)):
      if string[i] == "[":
        openBrackets += 1
      elif string[i] == "]":
        openBrackets -= 1
      elif string[i] == ",":
        if openBrackets == 0:
          item = string[lastComma:i]
          separated_items.append(item.strip())
          lastComma = i + 1
    if lastComma < len(string):
      item = string[lastComma:len(string)]
      separated_items.append(item.strip())
    return separated_items

  def _setBool(self, key, value):
    """Set value for a bool hyperparameter."""
    if value == "False":
      setattr(self, key, False)
    elif value == "True":
      setattr(self, key, True)
    else:
      raise ValueError("Expected True or False but received " + value)

  def _setInt(self, key, value):
    """Set value for an int hyperparameter."""
    try:
      setattr(self, key, int(value))
    except ValueError:
      raise ValueError("Expected an integer but received " + value)


  def _setFloat(self, key, value):
    """Set value for a float hyperparameter."""
    try:
      setattr(self, key, float(value))
    except ValueError:
      raise ValueError("Expected a float but received " + value)

  def _setList(self, key, value):
    """Set value of a list hyperparameter."""
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


