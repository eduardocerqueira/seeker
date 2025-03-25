#date: 2025-03-25T17:06:45Z
#url: https://api.github.com/gists/1a0c9ddfbc9b584c796ebcf1da85d0ff
#owner: https://api.github.com/users/RajChowdhury240

import boto3
import csv
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, TimeRemainingColumn, BarColumn, SpinnerColumn
from datetime import datetime

console = Console()
lock = threading.Lock()
session = boto3.session.Session()
org_client = session.client('organizations')

ROLE_NAME = 'ca-admin'
RESULTS = []

# Prefixes per resource type
LAMBDA_PREFIXES = ['dev-temp-', 'old-lambda-']
ALARM_PREFIXES = ['TestAlarm-', 'Deprecated-']
STACK_PREFIXES = ['demo-stack-', 'temp-']
STACKSET_PREFIXES = ['test-set-', 'legacy-']

def assume_role(account_id):
    sts = boto3.client('sts')
    role_arn = f'arn:aws:iam::{account_id}:role/{ROLE_NAME}'
    try:
        creds = sts.assume_role(RoleArn=role_arn, RoleSessionName='CleanupSession')['Credentials']
        return boto3.Session(
            aws_access_key_id= "**********"
            aws_secret_access_key= "**********"
            aws_session_token= "**********"
        )
    except Exception as e:
        console.log(f"[red]Failed to assume role in {account_id}: {e}")
        return None

def get_all_regions(service):
    try:
        return boto3.session.Session().get_available_regions(service)
    except Exception as e:
        console.log(f"[red]Failed to get regions for {service}: {e}")
        return []

def delete_lambda_functions(session, account_id, dry_run):
    deleted = []
    for region in get_all_regions('lambda'):
        client = session.client('lambda', region_name=region)
        try:
            paginator = client.get_paginator('list_functions')
            for page in paginator.paginate():
                for fn in page['Functions']:
                    name = fn['FunctionName']
                    if any(name.startswith(p) for p in LAMBDA_PREFIXES):
                        deleted.append((name, region))
                        if not dry_run:
                            client.delete_function(FunctionName=name)
        except Exception:
            continue
    return [('Lambda', name, region) for name, region in deleted]

def delete_cloudwatch_alarms(session, account_id, dry_run):
    deleted = []
    for region in get_all_regions('cloudwatch'):
        client = session.client('cloudwatch', region_name=region)
        try:
            alarms = client.describe_alarms()['MetricAlarms']
            matched = [alarm['AlarmName'] for alarm in alarms if any(alarm['AlarmName'].startswith(p) for p in ALARM_PREFIXES)]
            if matched:
                if not dry_run:
                    client.delete_alarms(AlarmNames=matched)
                for name in matched:
                    deleted.append((name, region))
        except Exception:
            continue
    return [('CloudWatchAlarm', name, region) for name, region in deleted]

def delete_cloudformation_stacks(session, account_id, dry_run):
    deleted = []
    for region in get_all_regions('cloudformation'):
        client = session.client('cloudformation', region_name=region)
        try:
            paginator = client.get_paginator('describe_stacks')
            for page in paginator.paginate():
                for stack in page['Stacks']:
                    name = stack['StackName']
                    status = stack['StackStatus']
                    if any(name.startswith(p) for p in STACK_PREFIXES) and not status.startswith('DELETE'):
                        deleted.append((name, region))
                        if not dry_run:
                            client.delete_stack(StackName=name)
        except Exception:
            continue
    return [('Stack', name, region) for name, region in deleted]

def delete_stacksets(session, account_id, dry_run):
    client = session.client('cloudformation')
    try:
        sets = client.list_stack_sets(Status='ACTIVE')['Summaries']
        matched = [s['StackSetName'] for s in sets if any(s['StackSetName'].startswith(p) for p in STACKSET_PREFIXES)]
        for name in matched:
            if not dry_run:
                try:
                    client.delete_stack_set(StackSetName=name)
                except Exception as e:
                    console.log(f"[yellow]Failed to delete StackSet {name} in {account_id}: {e}")
        return [('StackSet', name, 'global') for name in matched]
    except Exception:
        return []

def process_account(account_id, dry_run, progress_task, progress):
    session = assume_role(account_id)
    if not session:
        progress.advance(progress_task)
        return

    deleted_resources = []
    deleted_resources += delete_lambda_functions(session, account_id, dry_run)
    deleted_resources += delete_cloudwatch_alarms(session, account_id, dry_run)
    deleted_resources += delete_cloudformation_stacks(session, account_id, dry_run)
    deleted_resources += delete_stacksets(session, account_id, dry_run)

    with lock:
        for res_type, name, region in deleted_resources:
            RESULTS.append({
                "AccountID": account_id,
                "Type": res_type,
                "Name": name,
                "Region": region,
                "DryRun": dry_run
            })
    progress.advance(progress_task)

def main(dry_run):
    accounts = org_client.list_accounts()['Accounts']
    active_accounts = [acc['Id'] for acc in accounts if acc['Status'] == 'ACTIVE']

    with Progress(
        SpinnerColumn(), BarColumn(), "[progress.percentage]{task.percentage:>3.0f}%",
        "•", "[cyan]{task.description}", TimeRemainingColumn(), console=console
    ) as progress:
        task = progress.add_task("[green]Deleting resources...", total=len(active_accounts))

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(process_account, acc, dry_run, task, progress)
                       for acc in active_accounts]
            for _ in as_completed(futures):
                pass

    csv_file = f"deletion_results_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["AccountID", "Type", "Name", "Region", "DryRun"])
        writer.writeheader()
        writer.writerows(RESULTS)

    table = Table(title="AWS Resource Deletion Report")
    table.add_column("Account ID", style="cyan")
    table.add_column("Type", style="magenta")
    table.add_column("Name", style="green")
    table.add_column("Region", style="blue")
    table.add_column("DryRun", style="yellow")

    for r in RESULTS:
        table.add_row(r["AccountID"], r["Type"], r["Name"], r["Region"], str(r["DryRun"]))

    console.print(table)
    console.print(f"[bold green]Results saved to:[/bold green] {csv_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AWS Resource Cleaner Script")
    parser.add_argument('--dry-run', action='store_true', help='Only show what will be deleted')
    args = parser.parse_args()

    main(args.dry_run)gs()

    main(args.dry_run)