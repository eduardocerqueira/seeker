#date: 2025-05-16T16:49:44Z
#url: https://api.github.com/gists/d976ab2ee0d0be8164b51e248afb454c
#owner: https://api.github.com/users/RajChowdhury240

import boto3
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn
import time
from datetime import datetime

# Initialize AWS clients
org_client = boto3.client('organizations')
console = Console()

def get_nested_ous(parent_id):
    """Recursively fetch all nested OUs under a given parent ID."""
    ous = []
    paginator = org_client.get_paginator('list_organizational_units_for_parent')
    for page in paginator.paginate(ParentId=parent_id):
        for ou in page['OrganizationalUnits']:
            ous.append(ou)
            ous.extend(get_nested_ous(ou['Id']))
    return ous

def get_accounts_in_ou(ou_id):
    """Fetch all active accounts in a given OU."""
    accounts = []
    paginator = org_client.get_paginator('list_accounts_for_parent')
    for page in paginator.paginate(ParentId=ou_id):
        for account in page['Accounts']:
            if account['Status'] == 'ACTIVE':
                accounts.append(account)
    return accounts

def get_policy_details(policy_id):
    """Fetch SCP policy details by policy ID."""
    try:
        response = org_client.describe_policy(PolicyId=policy_id)
        policy = response['Policy']
        return {
            'PolicyName': policy['PolicySummary']['Name'],
            'PolicyContent': json.loads(policy['Content']),
            'PolicyId': policy_id
        }
    except org_client.exceptions.ClientError as e:
        console.log(f"[red]Error fetching policy {policy_id}: {e}[/red]")
        return None

def get_attached_policies(target_id, target_type):
    """Fetch all SCP policies attached to a target (account, OU, or root)."""
    policies = []
    paginator = org_client.get_paginator('list_policies_for_target')
    for page in paginator.paginate(TargetId=target_id, Filter='SERVICE_CONTROL_POLICY'):
        for policy in page['Policies']:
            policy_details = get_policy_details(policy['Id'])
            if policy_details:
                policies.append(policy_details)
    return policies

def process_account(account, root_id, ou_path_map):
    """Process an account to retrieve its SCP policies and OU hierarchy."""
    account_id = account['Id']
    account_name = account['Name']
    
    # Get account-level SCPs
    account_scps = get_attached_policies(account_id, 'ACCOUNT')
    
    # Get OU-level SCPs
    ou_scps = []
    current_ou_id = org_client.list_parents(ChildId=account_id)['Parents'][0]['Id']
    ou_path = ou_path_map.get(current_ou_id, f"OU_{current_ou_id}")
    
    while current_ou_id != root_id:
        ou_policies = get_attached_policies(current_ou_id, 'ORGANIZATIONAL_UNIT')
        ou_scps.extend(ou_policies)
        try:
            current_ou_id = org_client.list_parents(ChildId=current_ou_id)['Parents'][0]['Id']
        except:
            break
    
    # Get root-level SCPs
    root_scps = get_attached_policies(root_id, 'ROOT')
    
    return {
        'AccountId': account_id,
        'AccountName': account_name,
        'OUPath': ou_path,
        'AccountSCPs': account_scps,
        'OUSCPs': ou_scps,
        'RootSCPs': root_scps
    }

def main():
    start_time = time.time()
    output_data = []
    
    # Get organization root
    response = org_client.list_roots()
    root_id = response['Roots'][0]['Id']
    
    # Get all OUs recursively
    console.log("[blue]Fetching Organizational Units...[/blue]")
    ous = get_nested_ous(root_id)
    
    # Create OU path map
    ou_path_map = {root_id: "Root"}
    for ou in ous:
        path = []
        current_id = ou['Id']
        while current_id != root_id:
            parent = org_client.list_parents(ChildId=current_id)['Parents'][0]
            parent_id = parent['Id']
            parent_name = parent_id if parent_id == root_id else next((o['Name'] for o in ous if o['Id'] == parent_id), parent_id)
            path.append(parent_name)
            current_id = parent_id
        path.reverse()
        ou_path_map[ou['Id']] = "/".join(path) + f"/{ou['Name']}"
    
    # Get all accounts (root + all OUs)
    console.log("[blue]Fetching Accounts...[/blue]")
    all_accounts = get_accounts_in_ou(root_id)
    for ou in ous:
        all_accounts.extend(get_accounts_in_ou(ou['Id']))
    
    console.log(f"[green]Found {len(all_accounts)} active accounts.[/green]")
    
    # Process accounts with progress bar
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Processing accounts...", total=len(all_accounts))
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_account = {
                executor.submit(process_account, account, root_id, ou_path_map): account
                for account in all_accounts
            }
            
            for future in as_completed(future_to_account):
                account = future_to_account[future]
                progress.update(task, advance=1, description=f"Processing {account['Name']} ({account['Id']})")
                try:
                    result = future.result()
                    output_data.append(result)
                except Exception as e:
                    console.log(f"[red]Error processing account {account['Id']}: {e}[/red]")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"aws_scp_policies_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    console.log(f"[green]Output saved to {output_file}[/green]")
    console.log(f"[green]Total execution time: {time.time() - start_time:.2f} seconds[/green]")

if __name__ == "__main__":
    main()
