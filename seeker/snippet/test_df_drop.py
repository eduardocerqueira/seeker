#date: 2021-12-14T17:00:18Z
#url: https://api.github.com/gists/f29e8d2703789491e8e24e41de16536b
#owner: https://api.github.com/users/kozlov-alexey

import numba
from numba import njit
from numba.core import types

import sdc
from sdc.utilities.utils import sdc_overload_method
from sdc.functions import numpy_like
from sdc.hiframes.pd_dataframe_type import DataFrameType
from sdc.functions.tuple_utils import sdc_tuple_map
from sdc.hiframes.pd_dataframe_ext import init_dataframe_internal

import numpy as np
import pandas as pd


@njit
def drop_from_columns(array_data_list, drop_mask):

    res = []
    for arr in array_data_list:
        res.append(numpy_like.getitem_by_mask(arr, drop_mask))

    return res


@sdc_overload_method(DataFrameType, 'drop')
def pd_dataframe_drop_overload(df, labels):
    if not isinstance(df, DataFrameType):
        return None

    df_type = df

    def pd_dataframe_drop_impl(df, labels):
        print("DEBUG: Calling alternative df.drop impl!")
        df_len = len(df)
        res_drop_mask = np.zeros(df_len, dtype=types.bool_)
        df_index = df.index
        for i in numba.prange(df_len):
            if df_index[i] in labels:
                res_drop_mask[i] = True
        res_drop_mask = ~res_drop_mask

        res_df_data = sdc_tuple_map(
            drop_from_columns,
            df._data,
            res_drop_mask
        )
        res_df_index = df.index[res_drop_mask]
        return init_dataframe_internal(res_df_data, res_df_index, df_type)

    return pd_dataframe_drop_impl


@njit
def test_impl(df, labels):
    return df.drop(labels)


data = pd.DataFrame({'A': np.arange(10), 'B': np.ones(10)}, index=[0, 1, 2, 3, 1, 1, 0, 2, 3, 5])
print(test_impl(data, [0, 1]))

