#date: 2025-03-13T16:48:09Z
#url: https://api.github.com/gists/5b63162fcb898adcb8e8bb79099f80c0
#owner: https://api.github.com/users/RajChowdhury240

#!/usr/bin/env python3

import boto3
import botocore
import csv
import os
import sys
import time
from colorama import init, Fore, Style
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

init(autoreset=True)

# Hardcode the single account and role
ACCOUNT_ID = "111111111111"   # Replace with your real AWS Account ID
ROLE_NAME = "ca-admin"        # The role to assume
REGION_NAME = "us-east-1"     # Default region (adjust if needed)
OUTPUT_CSV = "deletion_log.csv"

def assume_role(account_id, role_name):
    """
    Assumes the specified role in the given AWS account and returns a temporary Session.
    """
    sts_client = boto3.client("sts")
    role_arn = f"arn:aws:iam::{account_id}:role/{role_name}"
    try:
        response = sts_client.assume_role(
            RoleArn=role_arn,
            RoleSessionName=f"AssumeRole-{account_id}",
            DurationSeconds=3600
        )
        creds = response["Credentials"]
        return boto3.session.Session(
            aws_access_key_id= "**********"
            aws_secret_access_key= "**********"
            aws_session_token= "**********"
            region_name=REGION_NAME
        )
    except botocore.exceptions.ClientError as e:
        print(f"{Fore.RED}Error assuming role in account {account_id}: {e}")
        return None

def any_matches(resource_name, patterns):
    """
    Returns True if 'resource_name' matches any wildcard pattern in 'patterns'.
    e.g. "Swap-Utilization-*" or "cfn-ec2-*".
    """
    import fnmatch
    for pat in patterns:
        if fnmatch.fnmatch(resource_name, pat):
            return True
    return False

def delete_cloudwatch_alarms(session, patterns):
    """
    Deletes CloudWatch alarms matching any prefix pattern in 'patterns'.
    """
    client = session.client("cloudwatch")
    alarms_to_delete = []
    paginator = client.get_paginator("describe_alarms")

    for page in paginator.paginate():
        for alarm in page["MetricAlarms"]:
            if any_matches(alarm["AlarmName"], patterns):
                alarms_to_delete.append(alarm["AlarmName"])

    if alarms_to_delete:
        try:
            client.delete_alarms(AlarmNames=alarms_to_delete)
            return alarms_to_delete
        except botocore.exceptions.ClientError as e:
            print(f"{Fore.RED}Failed to delete alarms: {e}")
    return []

def delete_lambda_functions(session, patterns):
    """
    Deletes Lambda functions matching any prefix pattern in 'patterns'.
    """
    client = session.client("lambda")
    deleted_functions = []
    paginator = client.get_paginator("list_functions")

    for page in paginator.paginate():
        for fn in page["Functions"]:
            fn_name = fn["FunctionName"]
            if any_matches(fn_name, patterns):
                try:
                    client.delete_function(FunctionName=fn_name)
                    deleted_functions.append(fn_name)
                except botocore.exceptions.ClientError as e:
                    print(f"{Fore.RED}Failed to delete Lambda function {fn_name}: {e}")
    return deleted_functions

def delete_cloudformation_stacks(session, patterns):
    """
    Deletes CloudFormation stacks matching any prefix pattern in 'patterns'.
    """
    client = session.client("cloudformation")
    deleted_stacks = []
    paginator = client.get_paginator("list_stacks")

    # Only consider stacks in stable states
    valid_stack_states = [
        "CREATE_COMPLETE", "UPDATE_COMPLETE", "ROLLBACK_COMPLETE",
        "UPDATE_ROLLBACK_COMPLETE", "IMPORT_COMPLETE"
    ]

    for page in paginator.paginate(StackStatusFilter=valid_stack_states):
        for summary in page["StackSummaries"]:
            stack_name = summary["StackName"]
            if any_matches(stack_name, patterns):
                try:
                    client.delete_stack(StackName=stack_name)
                    deleted_stacks.append(stack_name)
                except botocore.exceptions.ClientError as e:
                    print(f"{Fore.RED}Failed to delete stack {stack_name}: {e}")
    return deleted_stacks

def delete_cloudformation_stacksets(session, patterns):
    """
    Deletes CloudFormation stack sets matching any prefix pattern in 'patterns'.
    NOTE: If stack sets still have instances, you may need to remove them first.
    """
    client = session.client("cloudformation")
    deleted_stacksets = []

    paginator = client.get_paginator("list_stack_sets")
    for page in paginator.paginate(Status="ACTIVE"):
        for summary in page.get("Summaries", []):
            stackset_name = summary["StackSetName"]
            if any_matches(stackset_name, patterns):
                try:
                    client.delete_stack_set(StackSetName=stackset_name)
                    deleted_stacksets.append(stackset_name)
                except botocore.exceptions.ClientError as e:
                    print(f"{Fore.RED}Failed to delete stack set {stackset_name}: {e}")
    return deleted_stacksets

def cleanup_single_account(account_id, patterns):
    """
    Performs cleanup in the single AWS account:
      - Assumes the 'ca-admin' role
      - Deletes CloudWatch alarms, Lambda functions, CloudFormation stacks, and stack sets
    Returns a dict of what was deleted for logging.
    """
    result = {
        "AccountId": account_id,
        "DeletedAlarms": [],
        "DeletedLambdaFunctions": [],
        "DeletedStacks": [],
        "DeletedStackSets": []
    }

    session = assume_role(account_id, ROLE_NAME)
    if not session:
        return result  # if assume-role failed, nothing is deleted

    # Delete resources
    result["DeletedAlarms"] = delete_cloudwatch_alarms(session, patterns)
    result["DeletedLambdaFunctions"] = delete_lambda_functions(session, patterns)
    result["DeletedStacks"] = delete_cloudformation_stacks(session, patterns)
    result["DeletedStackSets"] = delete_cloudformation_stacksets(session, patterns)

    return result

def main(patterns_file):
    # Read patterns (like Swap-Utilization-*, cfn-ec2-*) from .txt
    if not os.path.exists(patterns_file):
        print(f"{Fore.RED}Error: Patterns file '{patterns_file}' not found.")
        sys.exit(1)

    with open(patterns_file, "r") as f:
        raw_patterns = [line.strip() for line in f if line.strip()]
        patterns = [p for p in raw_patterns if p]

    print(f"{Fore.CYAN}Starting cleanup in account {ACCOUNT_ID}...")

    start_time = time.time()

    # We'll log results in a CSV
    with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        # Write the header
        writer.writerow([
            "AccountId",
            "CloudWatchAlarms",
            "LambdaFunctions",
            "CloudFormationStacks",
            "CloudFormationStackSets"
        ])
        
        # Show a small progress bar for the single account
        for _ in tqdm(range(1), desc="Deleting resources", bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.RESET)):
            result = cleanup_single_account(ACCOUNT_ID, patterns)
            writer.writerow([
                result["AccountId"],
                ";".join(result["DeletedAlarms"]),
                ";".join(result["DeletedLambdaFunctions"]),
                ";".join(result["DeletedStacks"]),
                ";".join(result["DeletedStackSets"])
            ])

    end_time = time.time()
    elapsed = end_time - start_time

    print(f"\n{Fore.CYAN}Cleanup completed in {elapsed:.2f} seconds.")
    print(f"Summary log written to {OUTPUT_CSV}.\n")

    # Print summary in a colorful Rich table
    console = Console()
    table = Table(title="Deleted Resources Summary", show_lines=True)

    table.add_column("Account ID", style="cyan", justify="left")
    table.add_column("CloudWatch Alarms", style="magenta", justify="left")
    table.add_column("Lambda Functions", style="magenta", justify="left")
    table.add_column("CloudFormation Stacks", style="magenta", justify="left")
    table.add_column("CloudFormation StackSets", style="magenta", justify="left")

    table.add_row(
        result["AccountId"],
        str(len(result["DeletedAlarms"])),
        str(len(result["DeletedLambdaFunctions"])),
        str(len(result["DeletedStacks"])),
        str(len(result["DeletedStackSets"]))
    )

    console.print(table)

    # Optionally print more detailed list output
    print(f"{Style.BRIGHT}Detailed Deletions:{Style.RESET_ALL}")
    print(f"  - CloudWatch Alarms: {result['DeletedAlarms']}")
    print(f"  - Lambda Functions: {result['DeletedLambdaFunctions']}")
    print(f"  - CloudFormation Stacks: {result['DeletedStacks']}")
    print(f"  - CloudFormation StackSets: {result['DeletedStackSets']}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"{Fore.YELLOW}Usage: python {sys.argv[0]} <prefixes_file.txt>")
        sys.exit(1)

    patterns_file = sys.argv[1]
    main(patterns_file)gv[1]
    main(patterns_file)