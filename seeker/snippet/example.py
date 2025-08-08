#date: 2025-08-08T17:05:50Z
#url: https://api.github.com/gists/5666e4e9f54814a5f3fe9287b0fcbbcf
#owner: https://api.github.com/users/datavudeja

import numpy as np
import pandas as pd
from datetime import datetime, date
from decimal import Decimal
import uuid
from enum import Enum

class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3

class CustomClass:
    def __init__(self, name):
        self.name = name

# Example dictionary with various types
data = {
    'python_datetime': datetime(1990, 5, 15),
    'numpy_datetime': np.datetime64('2023-07-02'),
    'pandas_timestamp': pd.Timestamp('2023-07-03'),
    'complex_number': 1 + 2j,
    'numpy_complex': np.complex64(3 + 4j),
    'set_example': {1, 2, 3},
    'bytes_example': b'hello',
    'custom_class': CustomClass('example'),
    'enum_example': Color.RED,
    'uuid_example': uuid.uuid4(),
    'decimal_example': Decimal('3.14'),
    'numpy_array': np.array([1, 2, 3]),
    'pandas_series': pd.Series([4, 5, 6]),
    'pandas_dataframe': pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
}

# Convert to JSON
json_string = convert_to_json(data)
print(json_string)