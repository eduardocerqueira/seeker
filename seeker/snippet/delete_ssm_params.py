#date: 2024-05-20T17:11:49Z
#url: https://api.github.com/gists/4d1cb49f3ef21e7227cdc40f0ec92172
#owner: https://api.github.com/users/filipeandre

#!/usr/bin/env python
import boto3
import argparse


def delete_ssm_parameters(region, search_string):
    ssm_client = boto3.client('ssm', region_name=region)

    next_token = "**********"

    while True:
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"n "**********"e "**********"x "**********"t "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********": "**********"
            response = ssm_client.describe_parameters(
                NextToken= "**********"
            )
        else:
            response = ssm_client.describe_parameters()

        parameter_names = [param['Name'] for param in response['Parameters']]

        for name in parameter_names:
            if search_string in name:
                print(f"Deleting parameter: {name}")
                ssm_client.delete_parameter(Name=name)

 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"' "**********"N "**********"e "**********"x "**********"t "**********"T "**********"o "**********"k "**********"e "**********"n "**********"' "**********"  "**********"i "**********"n "**********"  "**********"r "**********"e "**********"s "**********"p "**********"o "**********"n "**********"s "**********"e "**********": "**********"
            next_token = "**********"
        else:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Delete SSM parameters by name containing a specified string')
    parser.add_argument('-r', '--region', type=str, help='AWS region')
    parser.add_argument('-s', '--search-string', type=str, help='String to search for in parameter names')

    args = parser.parse_args()
    delete_ssm_parameters(args.region, args.search_string)
