#date: 2025-09-26T17:01:54Z
#url: https://api.github.com/gists/e034a8a77488bf6b49fa0a75b4189d5e
#owner: https://api.github.com/users/RajChowdhury240

#!/usr/bin/env python3
"""
AWS IAM Role Policy Scanner
Scans multiple AWS accounts for IAM roles with customer-managed policies containing 'delegated'
"""

import boto3
import csv
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List, Tuple, Dict
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.console import Console
from rich.table import Table
from rich import print as rprint
import logging

# Configuration
ASSUME_ROLE_NAME = 'ca-iam-cie-engineer'
MAX_WORKERS = 10  # Adjust based on your rate limits
OUTPUT_FILE = f'iam_delegated_policies_scan_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
KEYWORD = 'delegated'

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

console = Console()

class IAMPolicyScanner:
    def __init__(self, master_session: boto3.Session):
        self.master_session = master_session
        self.org_client = master_session.client('organizations')
        self.results = []
        self.results_lock = threading.Lock()
        self.failed_accounts = []
        
    def get_active_accounts(self) -> List[Dict]:
        """Get all active AWS accounts from the organization"""
        try:
            rprint("[bold green]Fetching active AWS accounts from organization...[/bold green]")
            
            paginator = self.org_client.get_paginator('list_accounts')
            accounts = []
            
            for page in paginator.paginate():
                for account in page['Accounts']:
                    if account['Status'] == 'ACTIVE':
                        accounts.append({
                            'Id': account['Id'],
                            'Name': account['Name']
                        })
            
            rprint(f"[bold cyan]Found {len(accounts)} active accounts[/bold cyan]")
            return accounts
            
        except Exception as e:
            logger.error(f"Error fetching accounts: {str(e)}")
            raise
    
    def assume_role(self, account_id: str) -> boto3.Session:
        """Assume role in the target account"""
        try:
            sts_client = self.master_session.client('sts')
            role_arn = f'arn:aws:iam::{account_id}:role/{ASSUME_ROLE_NAME}'
            
            response = sts_client.assume_role(
                RoleArn=role_arn,
                RoleSessionName=f'IAMPolicyScan-{account_id}-{int(time.time())}'
            )
            
            credentials = response['Credentials']
            
            return boto3.Session(
                aws_access_key_id= "**********"
                aws_secret_access_key= "**********"
                aws_session_token= "**********"
            )
            
        except Exception as e:
            logger.error(f"Failed to assume role in account {account_id}: {str(e)}")
            raise
    
    def scan_account_iam_policies(self, account_id: str, account_name: str) -> List[Tuple[str, str, str]]:
        """Scan IAM roles in a specific account for policies with 'delegated' keyword"""
        account_results = []
        
        try:
            # Assume role in target account
            target_session = self.assume_role(account_id)
            iam_client = target_session.client('iam')
            
            # Get all IAM roles
            paginator = iam_client.get_paginator('list_roles')
            
            for page in paginator.paginate():
                for role in page['Roles']:
                    role_name = role['RoleName']
                    
                    try:
                        # Get attached customer-managed policies
                        policy_paginator = iam_client.get_paginator('list_attached_role_policies')
                        
                        for policy_page in policy_paginator.paginate(RoleName=role_name):
                            for policy in policy_page['AttachedPolicies']:
                                policy_arn = policy['PolicyArn']
                                policy_name = policy['PolicyName']
                                
                                # Check if it's a customer-managed policy and contains 'delegated'
                                if not policy_arn.startswith('arn:aws:iam::aws:policy/'):
                                    if KEYWORD.lower() in policy_name.lower():
                                        account_results.append((account_id, role_name, policy_name))
                                        logger.info(f"Found match in {account_id}: Role {role_name} -> Policy {policy_name}")
                    
                    except Exception as e:
                        logger.warning(f"Error scanning role {role_name} in account {account_id}: {str(e)}")
                        continue
            
            return account_results
            
        except Exception as e:
            logger.error(f"Failed to scan account {account_id} ({account_name}): {str(e)}")
            self.failed_accounts.append((account_id, account_name, str(e)))
            return []
    
    def scan_account_worker(self, account_info: Dict, progress: Progress, task_id) -> List[Tuple[str, str, str]]:
        """Worker function for threading"""
        account_id = account_info['Id']
        account_name = account_info['Name']
        
        progress.update(task_id, description=f"[cyan]Scanning: {account_name} ({account_id})[/cyan]")
        
        try:
            results = self.scan_account_iam_policies(account_id, account_name)
            
            # Thread-safe result storage
            with self.results_lock:
                self.results.extend(results)
            
            progress.update(task_id, advance=1)
            return results
            
        except Exception as e:
            logger.error(f"Worker failed for account {account_id}: {str(e)}")
            progress.update(task_id, advance=1)
            return []
    
    def save_results_to_csv(self):
        """Save results to CSV file"""
        try:
            with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write header
                writer.writerow(['AccountID', 'RoleName', 'AttachedPolicyName'])
                
                # Write results
                for result in sorted(self.results):
                    writer.writerow(result)
            
            rprint(f"[bold green]Results saved to: {OUTPUT_FILE}[/bold green]")
            
        except Exception as e:
            logger.error(f"Error saving CSV file: {str(e)}")
            raise
    
    def display_summary(self):
        """Display scan summary"""
        console.print("\n" + "="*80)
        console.print("[bold cyan]SCAN SUMMARY[/bold cyan]")
        console.print("="*80)
        
        # Results table
        if self.results:
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Account ID", style="cyan")
            table.add_column("Role Name", style="green")
            table.add_column("Policy Name", style="yellow")
            
            for account_id, role_name, policy_name in sorted(self.results):
                table.add_row(account_id, role_name, policy_name)
            
            console.print(table)
        else:
            rprint("[bold yellow]No roles found with customer-managed policies containing 'delegated'[/bold yellow]")
        
        # Statistics
        console.print(f"\n[bold green]Total matches found: {len(self.results)}[/bold green]")
        
        if self.failed_accounts:
            console.print(f"[bold red]Failed accounts: {len(self.failed_accounts)}[/bold red]")
            for account_id, account_name, error in self.failed_accounts:
                console.print(f"  • {account_name} ({account_id}): {error}")
    
    def run_scan(self):
        """Main scan execution"""
        start_time = time.time()
        
        try:
            # Get all active accounts
            accounts = self.get_active_accounts()
            
            if not accounts:
                rprint("[bold red]No active accounts found![/bold red]")
                return
            
            # Setup progress bar
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn(),
                console=console,
                transient=False
            ) as progress:
                
                task = progress.add_task("[green]Scanning accounts...", total=len(accounts))
                
                # Execute scan with threading
                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    # Submit all tasks
                    futures = {
                        executor.submit(self.scan_account_worker, account, progress, task): account 
                        for account in accounts
                    }
                    
                    # Wait for completion
                    for future in as_completed(futures):
                        account = futures[future]
                        try:
                            future.result()
                        except Exception as e:
                            logger.error(f"Thread failed for account {account['Id']}: {str(e)}")
            
            # Save results and display summary
            self.save_results_to_csv()
            self.display_summary()
            
            end_time = time.time()
            duration = end_time - start_time
            
            rprint(f"\n[bold green]Scan completed in {duration:.2f} seconds[/bold green]")
            
        except Exception as e:
            logger.error(f"Scan failed: {str(e)}")
            raise


def main():
    """Main function"""
    try:
        rprint("[bold blue]AWS IAM Role Policy Scanner[/bold blue]")
        rprint("[dim]Searching for IAM roles with customer-managed policies containing 'delegated'[/dim]\n")
        
        # Initialize AWS session (assumes AWS credentials are configured)
        session = boto3.Session()
        
        # Verify credentials
        try:
            sts = session.client('sts')
            identity = sts.get_caller_identity()
            rprint(f"[dim]Running as: {identity.get('Arn', 'Unknown')}[/dim]\n")
        except Exception as e:
            rprint(f"[bold red]Error verifying AWS credentials: {str(e)}[/bold red]")
            return
        
        # Initialize and run scanner
        scanner = IAMPolicyScanner(session)
        scanner.run_scan()
        
    except KeyboardInterrupt:
        rprint("\n[bold yellow]Scan interrupted by user[/bold yellow]")
    except Exception as e:
        rprint(f"[bold red]Fatal error: {str(e)}[/bold red]")
        logger.exception("Fatal error occurred")


if __name__ == "__main__":
    main()rred")


if __name__ == "__main__":
    main()