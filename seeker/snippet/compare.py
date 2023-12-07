#date: 2023-12-07T17:03:04Z
#url: https://api.github.com/gists/d2e1fc147b7ad0a6dfd686318cc9da57
#owner: https://api.github.com/users/jeromedockes

import numpy as np
import pandas as pd
import subprocess
import timeit
import pathlib

import polars as pl

import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils._encode import _unique


def time_functions(X, y, sklearn_unique, np_unique):
    encoder = OrdinalEncoder()
    number = 10
    res = min(timeit.repeat("encoder.fit(X)", number=number, globals=locals()))
    print(f"ordinal encoder fit: {res / number:.2e}")

    gb = HistGradientBoostingRegressor(categorical_features="from_dtype")
    res = min(timeit.repeat("gb.fit(X, y)", number=number, globals=locals()))
    print(f"gradient boosting fit: {res / number:.2e}")

    s = X["col_0"]
    values = np.asarray(s)
    # for fair comparison don't ask main to recompute np.asarray as it is done anyway
    col = values if branch == "main" else s
    col_name = "array" if branch == "main" else "series"
    res = min(timeit.repeat("sklearn_unique(col)", number=number, globals=locals()))
    print(f"_unique({col_name}): {res / number:.2e}")

    res = min(timeit.repeat("s.unique()", number=number, globals=locals()))
    print(f"series.unique(): {res / number:.2e}")

    res = min(
        timeit.repeat("np_unique(values)", number=number, globals=locals())
    )
    print(f"np.unique(): {res / number:.2e}")
    print()


branch = (
    subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True,
        cwd=pathlib.Path(sklearn.__file__).parent,
    )
    .stdout.strip()
    .decode("utf-8")
)
print(f"branch: {branch}\n")
rng = np.random.default_rng(0)
categories = [f"category_{i:0>2}" for i in range(20)]
x_values = rng.choice(categories, size=1_000_000)
y = rng.normal(size=x_values.size)
dtype = pl.Categorical
X = pl.DataFrame({"col_0": pl.Series(x_values, dtype=dtype)})
print("polars\n========")
time_functions(X, y, _unique, np.unique)
X = X.to_pandas()
print("pandas\n========")
time_functions(X, y, _unique, np.unique)
