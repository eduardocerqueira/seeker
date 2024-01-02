#date: 2024-01-02T16:52:44Z
#url: https://api.github.com/gists/4ef2072763da268b1b1953e102ad21f1
#owner: https://api.github.com/users/flying-sheep

from __future__ import annotations

import warnings


def has_doctest() -> None:
    """Doctest!
    
    >>> import warnings
    >>> warnings.warn("Hi!", FutureWarning)
    """


def test_warn(suppress_env) -> None:
    warnings.warn("Hi!", FutureWarning)
