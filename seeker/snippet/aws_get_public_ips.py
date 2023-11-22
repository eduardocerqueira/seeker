#date: 2023-11-22T16:58:14Z
#url: https://api.github.com/gists/46c9d2a7e13f3e656a0f2dcc3f46a033
#owner: https://api.github.com/users/ChristophShyper

#!/usr/bin/env python3

import boto3
import botocore.exceptions


"""
Script for counting current usage of public IPv4 across all accounts and regions.
Showing average spent starting January 2024
More info https://aws.amazon.com/blogs/aws/new-aws-public-ipv4-address-charge-public-ip-insights/

PREREQUISITES:
role_name - IAM role name which exist in all accounts and can be assumed by the current one
master_account_id - Id of account having current session
regions - list of regions used in the organization; specifing them speeds things up
"""


role_name = "OrganizationMaster"
master_account_id = '123456789012'
regions = ['eu-central-1', 'eu-west-1', 'eu-north-1']
hourly_cost = 0.005
organizations_client = boto3.client('organizations')
sts_client = boto3.client('sts')


def get_all_accounts():
    """
    Get all accounts in organization.
    :return: list of dicts with accounts' details
    """
    response = organizations_client.get_paginator('list_accounts').paginate(
        PaginationConfig={
            'MaxItems': 200,
            'PageSize': 20
        }
    )
    accounts = []
    for page in response:
        for acc in page['Accounts']:
            if acc['Status'] == 'ACTIVE':
                accounts.append(
                    {
                        'Id': acc['Id'],
                        'Name': acc['Name']
                    }
                )
    return sorted(accounts, key=lambda x: x['Name'])


def handler():
    count = 0
    ret = []
    for account in get_all_accounts():
        account_id = account['Id']
        for region in regions:
            try:
                if account_id != master_account_id:
                    # Assume the specified IAM role in the account
                    assumed_role = sts_client.assume_role(
                        RoleArn=f"arn:aws:iam::{account_id}:role/{role_name}",
                        RoleSessionName="CheckPublicIPs"
                    )
                    session = boto3.Session(
                        aws_access_key_id= "**********"
                        aws_secret_access_key= "**********"
                        aws_session_token= "**********"
                    )
                    ec2_client = session.client("ec2", region_name=region)
                else:
                    # Or in master
                    ec2_client = boto3.client("ec2", region_name=region)
                # Get results
                response = ec2_client.describe_network_interfaces(
                    Filters=[{"Name": "association.public-ip", "Values": ["*"]}]
                )
                interfaces = response.get("NetworkInterfaces", [])
                if len(interfaces) > 0:
                    print(f"Account {account['Name']}, region {region}: {len(interfaces)}")
                    ret_obj = {
                        'Id': account_id,
                        'Name': account['Name'],
                        'PublicIPs': []
                    }
                    count += len(interfaces)
                    for eni in interfaces:
                        ret_obj['PublicIPs'].append(eni['Association']['PublicIp'])
                    ret.append(ret_obj)
            except botocore.exceptions.ClientError as e:
                if e.response['Error']['Code'] != 'UnauthorizedOperation':
                    print(f"Error for account {account_id}: {str(e)}")
            except Exception as e:
                print(f"Error for account {account_id}: {str(e)}")
    print(f'Total: {count}')
    print(f'Av.Cost: {count * hourly_cost * 740} USD')
    # print(f'\nOut: {ret}')


# For running as a script
if __name__ == '__main__':
    handler()
    exit(0)
Out: {ret}')


# For running as a script
if __name__ == '__main__':
    handler()
    exit(0)
