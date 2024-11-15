#date: 2024-11-15T16:51:31Z
#url: https://api.github.com/gists/c2da30158856c0c5d4f7a52fd9cf4855
#owner: https://api.github.com/users/anyweez

import pyarrow
import pyarrow.parquet as pq
import sys

if len(sys.argv) != 2:
    print("Usage: python nan-checklist.py <parquet_file_path>")
    sys.exit(1)

filename = sys.argv[1]

print("--------------------------------------")
print("------- Parquet validity check -------")
print("--------------------------------------")

print()
print("Checking for columns that can't be parsed by pyarrow...")
print()

def print_column_report():
    # Read in the parquet file to extract the column list. We'll check each column
    # to see if it can be read by pyarrow.
    parquet_file = pq.ParquetFile(filename)
    column_names = parquet_file.schema_arrow.names

    # `good_columns` are columns that can be parsed and loaded into a pyarrow table
    good_columns: list[str] = []
    # `failed_columns` are columns that can't be parsed and loaded into a pyarrow table
    failed_colums: list[str] = []

    for column_name in column_names:
        try:
            # Read a subset of the parquet table, which should include all columns that've already
            # been parsed successfully + whatever column we're trying to read now.
            table = pq.read_table(filename, columns=good_columns + [column_name])

            # This is comparable to what Subsalt's software does internally, and where we believe
            # the error is coming from. The specific error we observed on 2024/11/13 was due to
            # a NaN (not a number) value in a column that doesn't support NaN values, specifically
            # an int64 column.
            pyarrow.Table.from_pylist(
                table.to_pandas().to_dict('records'),
                schema=table.schema
            )
        except Exception as e:
            # Catches all exceptions that occur when reading the parquet file in. The one we
            # saw in practice was:
            #  Could not convert nan with type float: tried to convert to int64
            #
            # We've been able to reproduce this by creating a parquet file with an int64 column
            # that has a NaN value in it.
            print(f"  > Failed to read column {column_name}")
            print(f"  > {e}")
            print()

            failed_colums.append(column_name)
            continue

        good_columns.append(column_name)

    print(f"Columns that passed: {good_columns}")
    print(f"Columns that failed to read: {failed_colums}")

print_column_report()
