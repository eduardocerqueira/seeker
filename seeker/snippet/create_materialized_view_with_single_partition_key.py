#date: 2022-01-03T17:05:45Z
#url: https://api.github.com/gists/c52bc97a3b93e1f2d38b655d7f5c15ae
#owner: https://api.github.com/users/caniko

def create_materialized_view_with_single_partition_key(
    keyspace_name, new_partition_key, session, model
):
    primary_keys = tuple(
        pk for pk in model._primary_keys.keys() if pk != new_partition_key
    )
    session.execute(
        query=(
            f"CREATE MATERIALIZED VIEW {keyspace_name}.{model.__name__}_by_{new_partition_key} AS"
            f"  SELECT * FROM {keyspace_name}.{model.__name__}"
            f"  WHERE {new_partition_key} IS NOT NULL {' '.join((f'AND {primary_key} IS NOT NULL' for primary_key in primary_keys))}"
            f"  PRIMARY KEY ({new_partition_key}, {', '.join(primary_keys)})"
            f"  WITH comment='Allow query by {new_partition_key} instead of neuron_id';"
        )
    )
