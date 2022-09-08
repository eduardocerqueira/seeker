#date: 2022-09-08T17:20:57Z
#url: https://api.github.com/gists/f4556cb83daebe2897aebc834df77349
#owner: https://api.github.com/users/xultaeculcis

# -*- coding: utf-8 -*-
import logging
import time
from functools import wraps
from typing import Callable


def timed(func: Callable) -> Callable:
    """This decorator prints the execution time for the decorated function."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        logging.info(f"{func.__qualname__} is running...")
        result = func(*args, **kwargs)
        end = time.time()
        logging.info(f"{func.__qualname__} ran in {(end - start):.4f}s")
        return result

    return wrapper
