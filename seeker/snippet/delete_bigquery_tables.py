#date: 2022-08-29T17:17:49Z
#url: https://api.github.com/gists/912f14c53d876955e434ce3d309c78f2
#owner: https://api.github.com/users/pandradepx

from functools import partial
from google.cloud import bigquery


def filter_tables(client: bigquery.Client, dataset: str, table_prefix: str = ""):
    _dataset = client.get_dataset(dataset)
    return set(filter(lambda table_obj: table_obj.table_id.startswith(table_prefix), bq_client.list_tables(_dataset)))


def delete_table(client: bigquery.Client, table):
    print(table.table_id)
    client.delete_table(table)


if __name__ == "__main__":
    bq_client = bigquery.Client()
    delete_table_partial = partial(delete_table, bq_client)
    list(map(delete_table_partial, filter_tables(bq_client, "ingestao_nepos", "_airbyte_tmp_")))
