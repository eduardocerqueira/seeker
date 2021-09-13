#date: 2021-09-13T17:16:24Z
#url: https://api.github.com/gists/1d26dcc795727b6d555adc0175ec8955
#owner: https://api.github.com/users/wyx0-xyz

from inspect import signature

def typecheck(fn):
  def wrapper(*args):
    annotations = fn.__annotations__

    if annotations != {}:
      parameters = signature(fn).parameters

      for i, parameter in enumerate(parameters):
        if parameter in annotations:
          expected_type = annotations[parameter]

          if not isinstance(args[i], expected_type):
            raise TypeError(f'Argument {parameter} of {fn.__name__} must be {expected_type}.')

      return fn(*args)

  return wrapper
