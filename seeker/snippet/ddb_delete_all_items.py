#date: 2021-10-26T16:55:28Z
#url: https://api.github.com/gists/f79d0d8865e5dced6f38dd53b7271da7
#owner: https://api.github.com/users/jwilson8767

import boto3

def ddb_delete_all_items(table_id, partition_key, sort_key=None):
    """
    Bulk deletion of all items in a DynamoDB Table.

    See also:
    [AWS REST API doc](https://docs.aws.amazon.com/amazondynamodb/latest/APIReference/API_BatchWriteItem.html)
    [boto3 doc](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/dynamodb.html#DynamoDB.Client.batch_write_item)

    :param table_id: The id of the table.
    :param partition_key: The name of the partition key
    :param sort_key: The name of the sort key for the table or None.
    """
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table(table_id)
    exclusive_start_key = {}
    to_delete = []
    while True:
        response = table.scan(
            **exclusive_start_key
        )
        for item in response['Items']:
            delete_key_dict = {
                partition_key: item[partition_key]
            }
            if sort_key is not None:
                delete_key_dict[sort_key] = item[sort_key]

            to_delete.append({'DeleteRequest': {
                'Key': delete_key_dict
            }})
        exclusive_start_key = {'ExclusiveStartKey': response['LastEvaluatedKey'] if 'LastEvaluatedKey' in response else None}
        if exclusive_start_key['ExclusiveStartKey'] is None:
            break
    failed_to_delete = []
    while len(to_delete):
        _to_delete = to_delete[:25]
        response = dynamodb.batch_write_item(
            RequestItems={table_id: _to_delete}
        )
        if 'UnprocessedItems' in response and table_id in response['UnprocessedItems']:
            failed_to_delete += response['UnprocessedItems'][table_id]

        to_delete = to_delete[25:]
    print(f'The following items could not be deleted from table "{table_id}": {failed_to_delete}')
