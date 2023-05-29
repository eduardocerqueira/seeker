#date: 2023-05-29T17:00:04Z
#url: https://api.github.com/gists/13a1c7f3e3734f3d27b4ab64b811795c
#owner: https://api.github.com/users/srikanth2310

import boto3

ec2 = boto3.resource('ec2')

def lambda_handler(event, context):
    # create filter for instances in running state
    filters = [
        {
            'Name': 'instance-state-name', 
            'Values': ['running']
        }
    ]
    
    # filter the instances based on filters() above
    instances = ec2.instances.filter(Filters=filters)

    # instantiate empty array
    RunningInstances = []

    for instance in instances:
        # for each instance, append to array and print instance id
        RunningInstances.append(instance.id)
        print instance.id
