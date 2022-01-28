#date: 2022-01-28T17:02:08Z
#url: https://api.github.com/gists/1ceb490312c59e4fb6e4bc15b57e9707
#owner: https://api.github.com/users/grantmwilliams

import pyarrow as pa
import pyarrow.parquet as pq

file_name_mapping = {
    pa.int32(): "int32",
    pa.uint32(): "uint32",
    pa.int64(): "int64",
    pa.uint64(): "uint64"
}

int_types = [pa.int32(), pa.uint32(), pa.int64(), pa.uint64()]

def write_file(pa_type, file_name):
    schema = pa.schema([
        pa.field("idx", pa.string()),
        pa.field("val", pa_type),
    ])

    table = pa.Table.from_pydict({
        'idx': ["A", "B", "C"],
        'val': [4, 5, 6],
    }, schema=schema)

    pq.write_table(table, file_name, compression='snappy')

def read_file(pa_type, file_name):
    with open(file_name, "rb") as fp:
        parquet_file = pq.ParquetFile(fp)
        col_type = parquet_file.schema_arrow.field("val").type
        print(f"pa_type: {pa_type} -- schema_type: {col_type}")
        print(parquet_file.schema.column(1))

print("-" * 40)
for pa_type in int_types:
    file_name = f"pyarrow_bug/data/{file_name_mapping[pa_type]}_file.snappy.parquet"
    write_file(pa_type, file_name)
    read_file(pa_type, file_name)
    print("-" * 40)