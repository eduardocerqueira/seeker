#date: 2023-10-13T17:06:12Z
#url: https://api.github.com/gists/35c982b371311fbf8816822bf3f9ce92
#owner: https://api.github.com/users/egormkn

from importlib.resources import files

import traitlets
from anywidget import AnyWidget


class LocalStorage(AnyWidget):
    _esm = files(__package__).joinpath("localstorage.js").read_text(encoding="utf-8")

    key = traitlets.Unicode().tag(sync=True)  # TODO: Add key validation
    value = traitlets.Unicode(allow_none=True, default_value=None).tag(sync=True)
