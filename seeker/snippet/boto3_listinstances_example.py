#date: 2025-04-14T17:11:03Z
#url: https://api.github.com/gists/ba26eab7df6c6ff8fb3b04eac31b406e
#owner: https://api.github.com/users/VishalJ4306

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
