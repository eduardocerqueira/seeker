#date: 2025-06-20T17:03:26Z
#url: https://api.github.com/gists/7f1c9feab723e21383234a5651d3df78
#owner: https://api.github.com/users/RajChowdhury240

import boto3
import csv
from rich.progress import Progress

# Initialize AWS Organizations client
org_client = boto3.client('organizations')


def get_all_ous(client):
    """
    Recursively gather all Organizational Units (OUs) under the root, including full path names.
    Returns a list of dicts: { 'Id': <OU_ID>, 'Name': <OU_NAME>, 'Path': <FULL_PATH> }
    """
    roots = client.list_roots()['Roots']
    all_ous = []

    def recurse(parent_id, path):
        paginator = client.get_paginator('list_organizational_units_for_parent')
        for page in paginator.paginate(ParentId=parent_id):
            for ou in page['OrganizationalUnits']:
                ou_path = f"{path}/{ou['Name']}"
                all_ous.append({'Id': ou['Id'], 'Name': ou['Name'], 'Path': ou_path})
                recurse(ou['Id'], ou_path)

    # Start from root OU
    root = roots[0]
    root_path = root['Name']
    all_ous.append({'Id': root['Id'], 'Name': root['Name'], 'Path': root_path})
    recurse(root['Id'], root_path)
    return all_ous


def main():
    # Fetch all OUs
    ous = get_all_ous(org_client)

    # Open CSV for writing
    with open('active_accounts.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'Account Name',
            'Account ID',
            'OU Name (Full recursive path)',
            'OU ID',
            'Tags'
        ])

        # Progress bar for OUs
        with Progress() as progress:
            task = progress.add_task("[green]Processing OUs...", total=len(ous))

            for ou in ous:
                # For each OU, list its accounts
                paginator = org_client.get_paginator('list_accounts_for_parent')
                for page in paginator.paginate(ParentId=ou['Id']):
                    for account in page['Accounts']:
                        if account['Status'] == 'ACTIVE':
                            # Retrieve tags for the account
                            tags_resp = org_client.list_tags_for_resource(
                                ResourceId=account['Id']
                            )
                            tags = tags_resp.get('Tags', [])
                            # Serialize tags as key=value pairs separated by semicolons
                            tags_str = ";".join(
                                f"{t['Key']}={t['Value']}" for t in tags
                            )

                            # Write row to CSV
                            writer.writerow([
                                account['Name'],
                                account['Id'],
                                ou['Path'],
                                ou['Id'],
                                tags_str
                            ])

                # Advance progress after each OU
                progress.advance(task)

    print("\n[bold green]Done! Active accounts saved to 'active_accounts.csv'.")


if __name__ == '__main__':
    main()