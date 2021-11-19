#date: 2021-11-19T17:14:21Z
#url: https://api.github.com/gists/da49bcd001508afca7de8b23c569a679
#owner: https://api.github.com/users/danyi1212

import inspect

from django.db.models import CharField
from django.utils.module_loading import import_string


class TypeField(CharField):
    description = "Python Type (as import_string)"

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('max_length', 1024)
        super().__init__(*args, **kwargs)

    def get_prep_value(self, value):
        if value is None:
            return None
        elif inspect.isclass(value):
            return ".".join((value.__module__, value.__name__))
        else:
            return ".".join((value.__class__.__module__, value.__class__.__name__))

    def from_db_value(self, value, expression, connection):
        if value is not None:
            return import_string(value)

    def to_python(self, value):
        if inspect.isclass(value):
            return value
        elif value is not None:
            return import_string(value)