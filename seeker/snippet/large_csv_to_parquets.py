#date: 2021-08-31T13:12:10Z
#url: https://api.github.com/gists/dc71e00b2a78dc03e683d1d447c18ee2
#owner: https://api.github.com/users/2minchul

from contextlib import contextmanager

import dask.dataframe as dd
import pyarrow  # noqa
from dask.distributed import Client, LocalCluster

"""
Don't let `total_memory_limit` exceed your memory.
This script will aborts, if runtime memory usage exceeds `total_memory_limit`
In my experience, it works for me:
Single parquet file size ~= `blocksize` / 4.5
`total_memory_limit` ~= (`blocksize` * 10) * `worker`
"""


@contextmanager
def get_client(worker=2, threads_per_worker=1, total_memory_limit='3GB'):
    cluster = LocalCluster(
        n_workers=worker,
        threads_per_worker=threads_per_worker,
        memory_limit=total_memory_limit,
    )
    client = Client(cluster)
    yield client
    client.close()
    cluster.close()


if __name__ == '__main__':
    csv_file = 'base.csv'
    out_dir = 'out'

    with open(csv_file, 'r') as f:
        header: str = next(f)
    headers = header.strip().split(',')

    with get_client():
        d_type = {i: str for i in range(len(headers))}  # modify it for your case
        df = dd.read_csv(csv_file, blocksize='150MB', dtype=d_type)
        df.to_parquet(out_dir, engine='pyarrow', compression='gzip')
