#date: 2021-11-10T17:09:53Z
#url: https://api.github.com/gists/e45164846dc7f93fce30846c86a79e82
#owner: https://api.github.com/users/wild-endeavor

import typing
from abc import ABC

import numpy as np
import pandas as pd
import pyarrow as pa

from flytekit import kwtypes, task, workflow
from flytekit.types.schema import SchemaFormat
from flytekit.types.structured.structured_dataset import (
    FLYTE_DATASET_TRANSFORMER,
    DatasetDecodingHandler,
    DatasetEncodingHandler,
    FlyteDataset,
    FlyteDatasetMetadata,
)

PANDAS_PATH = "/tmp/pandas.pq"  # should be randomly generated.
NUMPY_PATH = "/tmp/numpy.pq"
BQ_PATH = "bq://photo-313016:flyte.new_table3"


@task
def t0() -> pd.DataFrame:
    return pd.DataFrame({"Name": ["Tom", "Joseph"], "Age": [20, 22]})


@task
def t1(dataframe: pd.DataFrame) -> FlyteDataset[FlyteDatasetMetadata(columns=kwtypes(x=int, y=str), path=PANDAS_PATH)]:
    # Pandas -> S3 (parquet)
    return dataframe

@task
def t1(dataframe: pd.DataFrame) -> FlyteDataset[kwtypes(x=int, y=str)]:
    # Pandas -> S3 (parquet)
    #                                                                  s3://my-bucket/pq_file_shoud_be_here
    return FlyteDataset(dataframe=dataframe, meta=FlyteDatasetMetadata(remote_path=PANDAS_PATH))


# should trigger downloading and re-uploading
@task
def t2(dataframe: pd.DataFrame) -> pd.DataFrame:
    # Pandas -> Pandas
    return dataframe


@task
def t3(
    dataframe: FlyteDataset[FlyteDatasetMetadata(columns=kwtypes(x=int, y=str))]
) -> FlyteDataset[FlyteDatasetMetadata(columns=kwtypes(x=int, y=str))]:
    # s3 (parquet) -> pandas -> s3 (parquet)
    print("Pandas dataframe")
    print(dataframe.open_as(pd.DataFrame))  # should trigger download of data
    return dataframe  # should not trigger upload, since it's the same input object


# This task should not do anything - no uploading, no downloading
@task
def t3.5(ds: FlyteDataset[kwtypes(x=int, y=str)]) -> FlyteDataset[kwtypes(x=int, y=str)]:
    return ds

@task
def t4(dataframe: FlyteDataset[kwtypes(x=int, y=str)]) -> pd.DataFrame:
    # s3 (parquet) -> pandas -> s3 (parquet)
    return dataframe.open_as(pd.DataFrame)


@task
def t5(dataframe: pd.DataFrame) -> FlyteDataset[FlyteDatasetMetadata(columns=kwtypes(x=int, y=str), path=BQ_PATH)]:
    # pandas -> bq
    return dataframe


# Decide if bigquery is also a storage format, in addition to being a storage location
@task
def t6(
    dataframe: FlyteDataset[kwtypes(x=int, y=str), storage_fmt=DatasetStorageFormats.BIGQUERY)]
) -> pd.DataFrame:
    # pandas -> bq
    df = dataframe.open_as(pd.DataFrame)
    return df


@task
def t7(
    df1: pd.DataFrame, df2: pd.DataFrame
) -> (
    FlyteDataset[FlyteDatasetMetadata(columns=kwtypes(x=int, y=str))],
    FlyteDataset[FlyteDatasetMetadata(columns=kwtypes(x=int, y=str))],
):
    # df1: pandas -> bq
    # df2: pandas -> s3 (parquet)
    return df1, df2


@task
def t8(dataframe: pa.Table) -> FlyteDataset[FlyteDatasetMetadata(columns=kwtypes(x=int, y=str))]:
    # Arrow table -> s3 (parquet)
    print("Arrow table")
    print(dataframe.columns)
    return dataframe


class NumpyEncodingHandlers(DatasetEncodingHandler):
    # needs to say what it accepts
    #              what it produces e.g. T
    
    def encode(self, dataframe: np.ndarray, name: typing.Optional[typing.List[str]] = None) -> T:
        if name is None:
            name = ["col" + str(i) for i in range(len(dataframe))]
        return pa.Table.from_arrays(dataframe, name)


class NumpyDecodingHandlers(DatasetDecodingHandler, ABC):
    def decode(self, table: pa.Table):
        return table.to_pandas().to_numpy()


FLYTE_DATASET_TRANSFORMER.register_handler(np.ndarray, NumpyEncodingHandlers())
FLYTE_DATASET_TRANSFORMER.register_handler(np.ndarray, NumpyDecodingHandlers())

FLYTE_DATASET_TRANSFORMER.get_literal_type(my_arrow_schema) # what happens here?
# output would depend on the handlers registered with this transformer...

@task
def t9(dataframe: np.ndarray) -> FlyteDataset[FlyteDatasetMetadata(columns=kwtypes(x=int, y=str), path=NUMPY_PATH)]:
    # numpy -> Arrow table -> s3 (parquet)
    return dataframe


@task
def t10(dataframe: FlyteDataset[FlyteDatasetMetadata(columns=kwtypes(x=int, y=str), path=PANDAS_PATH)]) -> np.ndarray:
    # s3 (parquet) -> Arrow table -> numpy
    np_array = dataframe.open_as(np.ndarray)
    return np_array


# We see numpy as custom dataframe here
# we didn't implement a handler to R/W bigquery, but we still can R/W bigquery
@task
def t11(
    dataframe: np.ndarray,
) -> FlyteDataset[FlyteDatasetMetadata(columns=kwtypes(x=int, y=str), path="bq://photo-313016:flyte.new_table5")]:
    # numpy -> Arrow table -> bq
    return dataframe


@workflow()
def wf():
    df = t0()
    t7(df1=df, df2=df)


if __name__ == "__main__":
    data = {"Name": ["Tom", "Joseph"], "Age": [20, 22]}
    df = pd.DataFrame(data)
    np_array = np.array([[1, 2], [4, 5]])

    t1(dataframe=df)
    t2(dataframe=df)
    t3(dataframe=FlyteDataset(local_path=PANDAS_PATH))
    t4(dataframe=FlyteDataset(local_path=PANDAS_PATH))
    t5(dataframe=df)
    t6(dataframe=FlyteDataset(remote_path=BQ_PATH))
    t7(df1=df, df2=df)
    t8(dataframe=pa.Table.from_pandas(df))
    t9(dataframe=np_array)
    t10(dataframe=FlyteDataset(local_path=NUMPY_PATH))
    t11(dataframe=np_array)