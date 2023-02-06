#date: 2023-02-06T16:52:28Z
#url: https://api.github.com/gists/ff8cf79e835d7b68cb5b3182a90e84ea
#owner: https://api.github.com/users/mattiabertorello

import boto3
import json
client = boto3.client('config')

# If more than 100 add pagination
response = client.select_resource_config(
    Expression="""
    SELECT
        resourceId
    WHERE
      resourceType = 'AWS::EC2::Volume'
      AND tags.key <> 'aws:elasticmapreduce:instance-group-role'
      AND configuration.state.value <> 'in-use'
    """,
    Limit=100,
)
volume_ids = [json.loads(r)['resourceId'] for r in response['Results']]

ec2 = boto3.resource('ec2')
for volume_id in volume_ids:
    volume = ec2.Volume(volume_id)
    volume.delete()