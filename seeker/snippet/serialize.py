#date: 2025-08-08T17:05:50Z
#url: https://api.github.com/gists/5666e4e9f54814a5f3fe9287b0fcbbcf
#owner: https://api.github.com/users/datavudeja

import json
from datetime import datetime, date
import numpy as np
import pandas as pd
from decimal import Decimal
import uuid
from enum import Enum

def advanced_serializer(obj):
    # Datetime objects
    if isinstance(obj, (datetime, date)):
        return obj.strftime('%Y-%m-%d')
    
    # NumPy and Pandas date/time objects
    elif isinstance(obj, np.datetime64):
        return str(obj)[:10]
    elif isinstance(obj, np.timedelta64):
        return str(obj)
    elif isinstance(obj, pd.Timestamp):
        return obj.strftime('%Y-%m-%d')
    elif isinstance(obj, pd.Timedelta):
        return str(obj)
    
    # NumPy numeric types
    elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, 
                          np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    
    # Complex numbers
    elif isinstance(obj, (complex, np.complex64, np.complex128)):
        return str(obj)
    
    # Sets
    elif isinstance(obj, set):
        return list(obj)
    
    # Bytes
    elif isinstance(obj, bytes):
        return obj.decode('utf-8')
    
    # Custom classes
    elif hasattr(obj, '__dict__'):
        return obj.__dict__
    
    # Enum
    elif isinstance(obj, Enum):
        return obj.value
    
    # UUID
    elif isinstance(obj, uuid.UUID):
        return str(obj)
    
    # Decimal
    elif isinstance(obj, Decimal):
        return float(obj)
    
    # NumPy array
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    
    # Pandas Series
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    
    # Pandas DataFrame
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    
    # If we don't recognize the type, raise an error
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

# Example usage:
def convert_to_json(data):
    return json.dumps(data, default=advanced_serializer)