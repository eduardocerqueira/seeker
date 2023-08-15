#date: 2023-08-15T16:52:23Z
#url: https://api.github.com/gists/5607c7ab56f239476bb1a37f2c062e2d
#owner: https://api.github.com/users/layaramchandra9

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
