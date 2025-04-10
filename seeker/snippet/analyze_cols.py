#date: 2025-04-10T16:54:18Z
#url: https://api.github.com/gists/03bebb42af50f705965a6c088b9acf3e
#owner: https://api.github.com/users/juraj-m

import pandas as pd

def analyze_columns(df: pd.DataFrame):
    """
    Analyze the DataFrame columns and return a list of columns that are likely
    to be unhelpful for data analysis or modeling.

    Checks:
    - Duplicate columns (same values across all rows)
    - Columns with only nulls
    - Columns with only one unique value (including all-null + one-value cases)

    Returns:
    - List of column names suggested for removal
    """
    cols_to_remove = set()

    # 1. Find duplicate columns
    duplicate_columns = set()
    columns_checked = set()
    for col1 in df.columns:
        for col2 in df.columns:
            if col1 == col2 or col2 in columns_checked:
                continue
            if df[col1].equals(df[col2]):
                duplicate_columns.add(col2)
        columns_checked.add(col1)
    cols_to_remove.update(duplicate_columns)

    # 2. Columns with only nulls
    null_only_columns = df.columns[df.isnull().all()].tolist()
    cols_to_remove.update(null_only_columns)

    # 3. Columns with only one unique value (including all nulls + one value)
    single_value_columns = df.columns[df.nunique(dropna=False) <= 1].tolist()
    cols_to_remove.update(single_value_columns)

    return list(cols_to_remove)
