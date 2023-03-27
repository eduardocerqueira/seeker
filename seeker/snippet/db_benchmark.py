#date: 2023-03-27T17:07:27Z
#url: https://api.github.com/gists/dba8d1f876e1d601f530c0e8b16d5a85
#owner: https://api.github.com/users/BexTuychiev

import time

import datatable as dt
import pandas as pd
import polars as pl

# Define a DataFrame to store the results
results_df = pd.DataFrame(
    columns=["Function", "Library", "Runtime (s)"]
)


# Define a timer decorator
def timer(results: pd.DataFrame):
    """
    A decorator to measure the runtime of the passed function. It stores the runtime, the function name, and 
    the passed function's "library" parameter into the `results` DataFrame as a single row.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            elapsed_time = end_time - start_time
            results.loc[len(results)] = [
                func.__name__,
                kwargs["library"],
                elapsed_time,
            ]
            return result

        return wrapper

    return decorator


# Task 1: Reading CSVs
@timer(results_df)
def read_csv(path, library):
    if library == "pandas":
        return pd.read_csv(path, engine="pyarrow")
    elif library == "polars":
        return pl.read_csv(path)
    elif library == "datatable":
        return dt.fread(str(path))


# Task 2: Writing to CSVs
@timer(results_df)
def write_to_csv(df, path, library):
    if library == "pandas":
        df.to_csv(path, index=False)
    elif library == "polars":
        df.write_csv(path)
    elif library == "datatable":
        dt.Frame(df).to_csv(str(path))


# Task 3: Reading to Parquet
@timer(results_df)
def read_parquet(path, library):
    if library == "pandas":
        return pd.read_parquet(path, engine="pyarrow")
    elif library == "polars":
        return pl.read_parquet(path)
    elif library == "datatable":
        return None


# Task 4: Writing to Parquet
@timer(results_df)
def write_to_parquet(df, path, library):
    if library == "pandas":
        df.to_parquet(path, index=False)
    elif library == "polars":
        df.write_parquet(str(path))
    elif library == "datatable":
        # Not implemented
        return None
        
# Task 5: Sort
@timer(results_df)
def sort(df, column, library):
    if library == "pandas":
        return df.sort_values(column)
    elif library == "polars":
        return df.sort(column)
    elif library == "datatable":
        return df.sort(column)


# Task 6: Groupby
@timer(results_df)
def groupby(df, library):
    if library == "pandas":
        return df.groupby("education")["age"].mean()
    elif library == "polars":
        return df.groupby("education").agg(pl.mean("age"))
    elif library == "datatable":
        return df[:, dt.mean(dt.f.age), dt.by("education")]
        
from pathlib import Path

# Define the file paths
data = Path("data")
data_csv = data / "census_data.csv"
data_parquet = data / "census_data.parquet"


# libraries = ["pandas", "polars", "datatable"]
l = "datatable"

# Task 3/4
df = read_parquet(data_parquet, library=l)
write_to_parquet(df, data_parquet, library=l)

# Task 1/2
df = read_csv(data_csv, library=l)
write_to_csv(df, data_csv, library=l)

# Task 5/6
sort(df, "age", library=l)
groupby(df, library=l)

results_df.columns = ["Function", "Library", "Runtime (s)"]

results_df.replace(
    {
        "pandas": "Pandas PyArrow",
        "polars": "Polars",
        "datatable": "Data.table",
    },
    inplace=True,
)

results_df.to_csv("data/results.csv", index=False)

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

g = sns.catplot(
    data=results_df,
    kind="bar",
    x="Function",
    y="Runtime (s)",
    hue="Library",
)

g.legend.set_title("")
g.despine(left=True)

plt.xticks(rotation=45)
plt.xlabel("")
plt.title(
    "Comparison of three data manipulation libraries on 50M rows of data"
)

plt.savefig("benchmark.png")
plt.show()