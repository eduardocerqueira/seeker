#date: 2025-08-08T17:04:02Z
#url: https://api.github.com/gists/711b01094ed7259a4c9a79a5ab9fcad5
#owner: https://api.github.com/users/datavudeja

"""
Module containing a function to preserve leading zeroes (and other symbols such as "+", spaces and
parentheses in phone numbers and zip codes) when reading data with pandas.

WARNING: Use this caution, I most likely haven't covered all cases in the tests I wrote.
"""
import inspect
import io
import pandas as pd
import re
from loguru import logger # pip install loguru

# a test DataFrame
test_df = pd.DataFrame({'first_name':['John', 'Albert'],
                        'phone_number':'+491234567',
                        'nb_brothers':[0, 0],
                        'nb_sisters':[1, 0],
                        'nb_pets':[2,3],
                        'latitude':[0.87, 0.5],
                        'longitude':[55.45, 56.12],
                        'zip_code':['01234', '45675']}).convert_dtypes()

# regex to match decimal numbers starting with 0
# e.g. "0.", "0.1" or "0,", "0,1" for Germany
# see https://regex101.com/r/qYyRcL/1
re_decimals = re.compile(r'^0[.,]\d*$')

def keep_as_str(value):
    """
    My own rules to determine whether I want a value to be converted
    to numerical or I want it kept as a string.
    This will be applied to the numerical columns of a pandas DataFrame
    obtained from a method such as read_csv or read_excel in order
    to avoid converting zip codes such as "0123" to "1234" or removing
    plus signs and symbols (spaces, parentheses...) from phone numbers.
    Parameters
    ----------
    value : any
    Examples
    --------
    >>> keep_as_str('0')
    False
    >>> keep_as_str('+0123')
    True
    >>> keep_as_str('0123')
    True
    >>> keep_as_str('0.123')
    False
    >>> keep_as_str('+132')
    True
    >>> keep_as_str(123)
    False
    """
    # assume value is a scalar taken from a Series that pandas
    # infered as numerical that we reread as str
    # so value should be a str with digits or null
    if isinstance(value, str):
        # check values starting with 0
        if value != '0' and value.startswith('0'):
            # if it's a decimal value then it's fine
            return not re_decimals.search(value)
        # case of phone numbers
        elif any(c in value for c in ('(', ')', '/')) or value.startswith('+'):
            return True
        # there is a minus but it's not at the start? also probably a phone number or zip code
        elif len(value) > 1 and '-' in value[1:]:
            return True
        # most likely a regular number (123, .123, 0.123, ...)
        else:
            return False
    # most likely a null (e.g. pd.NA or np.nan)
    else:
        return False

def read_with_custom_type_infer(read_method, **kwargs):
    """
    Uses provided `read_method` (e.g. pd.read_excel) and reinfers
    data types of columns pandas infered as numericals to avoid
    problems such as missing leading zeroes in zip codes.
    WARNING: when reading in chunks, you may end up with columns
    mixing str and numeric values ("object" data type).
    Notes
    -----
    It does this by reading the data once, then rereading numerical
    columns as string (yes it's dumb but I found no other way) and
    using the function `keep_as_str` on all values until one matches
    (i.e. a any() logic).
    Parameters
    ----------
    read_method : pandas.read_csv, pandas.read_excel, ...
        I only tested pandas.read_csv and pandas.read_excel. I assume
        there are others that will work
    **kwargs
        Keyword arguments to the pandas reading method
    Examples
    --------
    >>> import pandas as pd
    >>>
    >>> # let's save a test DataFrame to csv and Excel
    >>> test_df.to_csv('test.csv', index=False)
    >>> test_df.to_excel('test.xlsx', index=False)
    >>>
    >>> # read the file "normally", zip codes and phone numbers are incorrect
    >>> df = pd.read_csv(filepath_or_buffer='test.csv', sep=',')
    >>> print(df.to_markdown())
    |    | first_name   |   phone_number |   nb_brothers |   nb_sisters |   nb_pets |   latitude |   longitude |   zip_code |
    |---:|:-------------|---------------:|--------------:|-------------:|----------:|-----------:|------------:|-----------:|
    |  0 | John         |      491234567 |             0 |            1 |         2 |       0.87 |       55.45 |       1234 |
    |  1 | Albert       |      491234567 |             0 |            0 |         3 |       0.5  |       56.12 |      45675 |
    >>>
    >>> # and now with this function
    >>> df = read_with_custom_type_infer(read_method=pd.read_csv, filepath_or_buffer='test.csv', sep=',')
    >>> print(df.to_markdown())
    |    | first_name   |   phone_number |   nb_brothers |   nb_sisters |   nb_pets |   latitude |   longitude |   zip_code |
    |---:|:-------------|---------------:|--------------:|-------------:|----------:|-----------:|------------:|-----------:|
    |  0 | John         |     +491234567 |             0 |            1 |         2 |       0.87 |       55.45 |      01234 |
    |  1 | Albert       |     +491234567 |             0 |            0 |         3 |       0.5  |       56.12 |      45675 |
    >>>
    >>> # same with an Excel file
    >>> df = read_with_custom_type_infer(read_method=pd.read_excel, io='test.xlsx')
    >>> print(df.to_markdown())
    |    | first_name   |   phone_number |   nb_brothers |   nb_sisters |   nb_pets |   latitude |   longitude |   zip_code |
    |---:|:-------------|---------------:|--------------:|-------------:|----------:|-----------:|------------:|-----------:|
    |  0 | John         |     +491234567 |             0 |            1 |         2 |       0.87 |       55.45 |      01234 |
    |  1 | Albert       |     +491234567 |             0 |            0 |         3 |       0.5  |       56.12 |      45675 |
    """
    # dealing with that now would just complicate operations, so let's remove this
    index_col = kwargs.pop('index_col', None)
    df = read_method(**kwargs)

    # restart buffer
    io_args = ('io', 'filepath_or_buffer', 'path_or_buf') # this should cover most pd.read_x methods
    buffer = next((v for k, v in kwargs.items() if k in io_args), None)
    if isinstance(buffer, (io.BytesIO, io.StringIO)):
        buffer.seek(0)

    # local helpers
    logs = []
    def log_once(msg):
        if msg not in logs:
            logger.info(msg)
            logs.append(msg)

    def retrieve_strings(df, df_str):        
        # find columns pandas infered as numericals
        numeric_columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]

        # see if we want to retrieve the string "version" of the column with our rules
        for col in numeric_columns:
            if df_str[col].map(keep_as_str, na_action='ignore').fillna(False).any():
                log_once(f'Leading zero or plus sign detected in column "{col}", values will be read as str.')
                # handle case where column has been set as index
                if col in df.index.names:
                    # not the most proper method to do it but it works...should be something with
                    # df.index = df.index.set_levels(...) I suppose. Also it's not safe if there are duplicates amongst
                    # columns+index levels but you'd have problems before this step anyways
                    ix_names = df.index.names
                    df = df.reset_index().assign(col=df_str[col]).set_index(ix_names)
                elif col in df.columns:
                    df[col] = df_str[col]
                else:
                    raise AssertionError
        if index_col:
            df.set_index(index_col, inplace=True)
        return df

    # handle iterators
    if 'chunksize' not in kwargs or not kwargs['chunksize']:
        logger.info('Reinfering data types by rereading the data or if possible parts '
                    'of the data (columns infered by pandas as numericals)')
        # use usecols if possible (pd.read_json does not have it) to not reread everything
        if 'usecols' in inspect.signature(read_method).parameters:
            # reread only numerical columns
            kwargs['usecols'] = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        df_str = read_method(dtype=str, **kwargs)
        return retrieve_strings(df=df, df_str=df_str)
    else:
        # we can't know the numerical columns before reading all the data so
        # we cannot do `usecols=numeric_columns` here
        df_iter = df
        df_str_iter = read_method(dtype=str, **kwargs)
        def retrieve_strings_iter():
            for chunk, chunk_str in zip(df_iter, df_str_iter):
                yield retrieve_strings(chunk, chunk_str)
        return retrieve_strings_iter()