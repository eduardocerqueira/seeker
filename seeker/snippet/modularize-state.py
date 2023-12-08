#date: 2023-12-08T16:52:50Z
#url: https://api.github.com/gists/8269483cba2e78523306b7f82de147ee
#owner: https://api.github.com/users/ebuildy

#!/usr/bin/env python3

"""
To generate json plan:

terraform plan -var-file=..... -out=output.tfplan
terraform show -json output.tfplan > plan.json
"""

import argparse
import json
import sys

parser = argparse.ArgumentParser(
                    prog='modularize-state',
                    description='Generate terraform mv instructions to migrate state resources into module',
                    usage="bin/modularize-state.py plan.json, then copy paste intructions")

parser.add_argument('-v', '--verbose', action='store_true')
parser.add_argument('inplan', nargs='?', type=argparse.FileType('r'), default=sys.stdin)

args = parser.parse_args()

plan_data = json.loads(args.inplan.read())

resource_changes = plan_data["resource_changes"]

resources_to_delete = list(filter(lambda c: c["change"]["actions"][0] == "delete", resource_changes))
resources_to_create = list(filter(lambda c: c["change"]["actions"][0] == "create", resource_changes))

if args.verbose:
    [print(f"- {res['address']}") for res in resources_to_delete]
    print()
    [print(f"+ {res['address']}") for res in resources_to_create]
    print()

for resource_to_delete in resources_to_delete:
    resource_to_delete_id = resource_to_delete["address"]

    for resource_to_create in resources_to_create:
        resource_to_create_id = resource_to_create["address"]

        if resource_to_create_id.endswith(resource_to_delete_id):
            print(f"terraform state mv '{resource_to_delete_id}' '{resource_to_create_id}'")
