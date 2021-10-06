#date: 2021-10-06T17:10:04Z
#url: https://api.github.com/gists/0fd4d82cbb2bae207c91d6cf325694b0
#owner: https://api.github.com/users/dldx

import os

import inspect
import hashlib
import pandas as pd

class DataFrameCache(object):
    """Cache the results of a function that returns a dataframe by storing results as parquet files.

    Usage:
    cache = DataFrameCache(cache_folder="/path/to/cache/folder")

    @cache
    def slow_function(arg1, arg2,...):
        pass

    """

    def __init__(self, cache_folder):
        self.cache_folder = os.path.expanduser(cache_folder)

    def stringify_args(self, func, args, kwargs):
        """Convert awkward arguments (currently only dataframes) to strings"""
        args = list(args)
        for i, arg in enumerate(args):
            if type(arg) == pd.core.frame.DataFrame:
                args[i] = arg.to_csv()

        for key in kwargs:
            if type(kwargs[key]) == pd.core.frame.DataFrame:
                kwargs[key] = kwargs[key].to_csv()

        # Stringify default arguments in function too
        default_args = inspect.signature(func)

        return str((default_args, tuple(args), kwargs))

    def __call__(self, func):
        self.function_name = func.__name__
        print("Caching results of " + self.function_name)

        def wrapper(*args, **kwargs):
            # Remove optional decorator argument from method keyword arguments
            try:
                skip_caching = kwargs["dataframecache_skip"]
            except KeyError:
                skip_caching = False
            kwargs.pop("dataframecache_skip", None)
            try:
                if skip_caching:
                    raise ValueError
                else:
                    return self.read_record_from_cache(self.stringify_args(func, args, kwargs))
            except (OSError, ValueError):
                results_df = func(*args, **kwargs)
                self.write_record_to_cache(self.stringify_args(func, args, kwargs), results_df)
                return results_df

        return wrapper

    def read_record_from_cache(self, key):
        key_hash = hashlib.sha1(str.encode(key)).hexdigest()

        value_df = pd.read_parquet("{0}/{1}.pq".format(self.cache_folder, key_hash))
        print("Read record {0} from {1}/{2}.pq".format(key, self.cache_folder, key_hash))
        return value_df

    def write_record_to_cache(self, key, value_df):
        os.makedirs(self.cache_folder, exist_ok=True)

        key_hash = hashlib.sha1(str.encode(key)).hexdigest()
        value_df.to_parquet("{0}/{1}.pq".format(self.cache_folder, key_hash))
        print("Wrote record {0} to {1}/{2}.pq".format(key, self.cache_folder, key_hash))