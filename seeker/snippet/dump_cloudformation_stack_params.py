#date: 2024-05-20T16:57:28Z
#url: https://api.github.com/gists/1ef5f7e8e63d696ff1bafc7dee4be3a1
#owner: https://api.github.com/users/filipeandre

#!/usr/bin/env python
import boto3
import json
import argparse


def get_stacks(client, prefix, suffix):
    paginator = client.get_paginator('describe_stacks')
    page_iterator = paginator.paginate()

    filtered_stacks = []
    for page in page_iterator:
        for stack in page['Stacks']:
            if ('RootId' not in stack and
                    stack['StackStatus'] in ['CREATE_COMPLETE', 'UPDATE_COMPLETE'] and
                    stack['StackName'].startswith(prefix) and
                    stack['StackName'].endswith(suffix)):
                filtered_stacks.append(stack)

    return filtered_stacks


def get_stack_parameters(client, stack_name):
    stack_details = client.describe_stacks(StackName=stack_name)
    parameters = stack_details['Stacks'][0]['Parameters']

    return parameters


def write_parameters_to_json(stack_name, parameters):
    parameters_dict = {
        "Parameters": [
            {
                "ParameterKey": param['ParameterKey'],
                "ParameterValue": param['ParameterValue']
            } for param in parameters
        ]
    }

    file_name = f"{stack_name}_parameters.json"

    with open(file_name, 'w') as json_file:
        json.dump(parameters_dict, json_file, indent=4)


def main(region, prefix, suffix):
    client = boto3.client('cloudformation', region_name=region)
    stacks = get_stacks(client, prefix, suffix)

    for stack in stacks:
        stack_name = stack['StackName']
        print(f"Processing Stack: {stack_name}")
        parameters = get_stack_parameters(client, stack_name)
        write_parameters_to_json(stack_name, parameters)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process CloudFormation stacks and create JSON parameter files.")
    parser.add_argument("-r", '--region', required=True, help="The AWS region where the stacks are located")
    parser.add_argument("-p", '--prefix', required=True, help="The prefix to filter the stack names (e.g., -p='params-')")
    parser.add_argument("-s", '--suffix', required=True, help="The suffix to filter the stack names (e.g., -s='-dev2')")
    args = parser.parse_args()

    main(args.region, args.prefix, args.suffix)
