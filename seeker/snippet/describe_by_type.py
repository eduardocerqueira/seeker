#date: 2025-08-08T17:02:41Z
#url: https://api.github.com/gists/829684aebfd22c3dc865904b3811b284
#owner: https://api.github.com/users/datavudeja

import numpy as np
import pandas as pd
from pandas.core.dtypes.common import (
    is_numeric_dtype, is_datetime64_dtype, is_bool_dtype
)
from pandas.core.indexes.datetimes import DatetimeIndex


def describe_by_type(dataframe):
    """
    prints descriptions of all columns (grouped by numeric, datetime, boolean,
    and others) and DatetimeIndex (if any)
    :param dataframe: a pandas DataFrame
    :return: None
    """
    boolean, numeric, datetime, other = False, False, False, False
    for column in dataframe.columns:
        if is_bool_dtype(dataframe[column]):
            boolean = True
        elif is_numeric_dtype(dataframe[column]):
            numeric = True
        elif is_datetime64_dtype(dataframe[column]):
            datetime = True
        else:
            other = True

    # describe datetime columns and DatetimeIndex (if any)
    if isinstance(dataframe.index, DatetimeIndex):
        print(pd.Series(dataframe.index).describe())
        print('\n')

    if datetime:
        print(dataframe.describe(include=['datetime']))
        print('\n')

    # describe numeric columns (if any)
    if numeric:
        print(dataframe.describe())
        print('\n')

    # describe boolean columns (if any)
    if boolean:
        print(dataframe.describe(include=[np.bool]))
        print('\n')

    # describe other columns (if any)
    if other:
        print(dataframe.describe(exclude=[np.number, np.datetime64, np.bool]))
        print('\n')