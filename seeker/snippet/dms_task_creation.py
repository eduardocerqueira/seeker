#date: 2023-03-16T17:05:10Z
#url: https://api.github.com/gists/c26f79d3b55f58d75a091c3e29a0a35c
#owner: https://api.github.com/users/jrrincon

#identifier for the dms task
replication_task_identifier = f'rds-archival-{ARCHIVE_SCHEMA}-{archive_table.replace("_", "-")}-archive-task'

#definition of task settings
replication_task_settings = {
        "TargetMetadata": {
            "SupportLobs": True,
            "LimitedSizeLobMode": True,
            "LobMaxSize": 4,
            "LobChunkSize": 64
        },
        "FullLoadSettings": {
            "MaxFullLoadSubTasks": 4,
            "CommitRate": 25000
        },
        "Logging": {
            "EnableLogging": True
        }
    }

#definition of mapping rule
def copy_table_rule(id, name, schema, table):
    return {
        "rule-type": "selection",
        "rule-id": f'{id}',
        "rule-name": f'{name}',
        "object-locator": {
            "schema-name": f'{schema}',
            "table-name": f'{table}'
        },
        "rule-action": "explicit"
    }    

#parameters for mapping rule
table_mapping = {
    "rules": [
        copy_table_rule(1, "archive-table-selector", ARCHIVE_SCHEMA, archive_table)
    ]
}

#dms task creation
task = dms_client.create_replication_task(
                ReplicationTaskIdentifier=replication_task_identifier,
                SourceEndpointArn=SOURCE_ENDPOINT_ARN,
                TargetEndpointArn=TARGET_ENDPOINT_ARN,
                ReplicationInstanceArn=REPLICATION_INSTANCE_ARN,
                ReplicationTaskSettings=json.dumps(replication_task_settings),
                MigrationType='full-load',
                TableMappings=json.dumps(table_mapping),
                Tags=tags
            )['ReplicationTask']