#date: 2023-11-06T17:04:25Z
#url: https://api.github.com/gists/a4d166f6ba2bda4a98e0aefdd2abe6c2
#owner: https://api.github.com/users/sanjivesanjive

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
