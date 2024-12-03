#date: 2024-12-03T17:13:01Z
#url: https://api.github.com/gists/457f707a8ae41d5674a93ce62a820785
#owner: https://api.github.com/users/RajChowdhury240

import boto3
import csv
import threading
from concurrent.futures import ThreadPoolExecutor
from rich.progress import Progress
from rich.table import Table
from rich.console import Console

console = Console()

output_file = "ta-ca-roles-info.csv"
role_prefixes = ["ta-", "ca-"]

def assume_role(account_id, role_name):
    sts_client = boto3.client("sts")
    try:
        response = sts_client.assume_role(
            RoleArn=f"arn:aws:iam::{account_id}:role/{role_name}",
            RoleSessionName="TrustRelationshipAnalysis"
        )
        credentials = response["Credentials"]
        return boto3.client(
            "iam",
            aws_access_key_id= "**********"
            aws_secret_access_key= "**********"
            aws_session_token= "**********"
        )
    except Exception as e:
        console.print(f"[red]Failed to assume role in account {account_id}: {e}")
        return None

def list_roles(client):
    roles = []
    try:
        paginator = client.get_paginator("list_roles")
        for page in paginator.paginate():
            roles.extend(page["Roles"])
    except Exception as e:
        console.print(f"[red]Error listing roles: {e}")
    return roles

def analyze_trust_relationships(account_id, account_name, iam_client, result_lock, results):
    roles = list_roles(iam_client)
    ta_roles = [role for role in roles if role["RoleName"].startswith("ta-")]
    ca_roles = [role for role in roles if role["RoleName"].startswith("ca-")]

    for ca_role in ca_roles:
        ca_role_name = ca_role["RoleName"]
        trust_policy = ca_role.get("AssumeRolePolicyDocument", {})
        for statement in trust_policy.get("Statement", []):
            if statement.get("Effect") == "Allow":
                principals = statement.get("Principal", {})
                aws_principals = principals.get("AWS", [])
                if not isinstance(aws_principals, list):
                    aws_principals = [aws_principals]

                for principal in aws_principals:
                    if principal.startswith(f"arn:aws:iam::{account_id}:role/ta-"):
                        with result_lock:
                            results.append([
                                account_id,
                                account_name,
                                principal.split("/")[-1],
                                ca_role_name,
                                str(statement),
                            ])

def get_all_accounts():
    org_client = boto3.client("organizations")
    accounts = []
    try:
        paginator = org_client.get_paginator("list_accounts")
        for page in paginator.paginate():
            accounts.extend(page["Accounts"])
    except Exception as e:
        console.print(f"[red]Error fetching accounts: {e}")
    return accounts

def main():
    accounts = get_all_accounts()
    results = []
    result_lock = threading.Lock()

    with open(output_file, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["AccountID", "AccountName", "RoleName-With-Ta", "RoleName-With-Ca", "Trust relationship reference"])

    with Progress() as progress:
        task = progress.add_task("Analyzing roles...", total=len(accounts))

        def process_account(account):
            account_id = account["Id"]
            account_name = account["Name"]
            iam_client = assume_role(account_id, "ca-iam-cie-engineer")
            if iam_client:
                analyze_trust_relationships(account_id, account_name, iam_client, result_lock, results)
            progress.update(task, advance=1)

        with ThreadPoolExecutor(max_workers=10) as executor:
            for account in accounts:
                executor.submit(process_account, account)

    with open(output_file, "a", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(results)

    table = Table(title="Trust Relationships")
    table.add_column("AccountID")
    table.add_column("AccountName")
    table.add_column("RoleName-With-Ta")
    table.add_column("RoleName-With-Ca")
    table.add_column("Trust relationship reference")

    for result in results:
        table.add_row(*result)

    console.print(table)

if __name__ == "__main__":
    main()
(table)

if __name__ == "__main__":
    main()
