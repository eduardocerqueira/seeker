#date: 2025-09-26T17:03:22Z
#url: https://api.github.com/gists/3b92040e4d016e6072dc1aa8fbbd01e3
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
    
    def display_rich_table(self):
        """Display results in rich table format"""
        console.print("\n" + "="*100)
        console.print("[bold cyan]IAM ROLES WITH DELEGATED POLICIES - DETAILED RESULTS[/bold cyan]")
        console.print("="*100)
        
        if self.results:
            # Create main results table
            table = Table(
                show_header=True, 
                header_style="bold magenta",
                title=f"[bold green]Found {len(self.results)} IAM Roles with 'Delegated' Policies[/bold green]",
                title_style="bold green",
                border_style="bright_blue",
                show_lines=True
            )
            
            table.add_column("Account ID", style="cyan", width=12, justify="left")
            table.add_column("Role Name", style="green", min_width=20, justify="left")
            table.add_column("Attached Policy Name", style="yellow", min_width=30, justify="left")
            
            # Sort results by Account ID, then Role Name, then Policy Name
            sorted_results = sorted(self.results, key=lambda x: (x[0], x[1], x[2]))
            
            for account_id, role_name, policy_name in sorted_results:
                table.add_row(account_id, role_name, policy_name)
            
            console.print(table)
            
            # Account summary table
            account_summary = {}
            for account_id, role_name, policy_name in self.results:
                if account_id not in account_summary:
                    account_summary[account_id] = {'roles': set(), 'policies': set(), 'total_matches': 0}
                account_summary[account_id]['roles'].add(role_name)
                account_summary[account_id]['policies'].add(policy_name)
                account_summary[account_id]['total_matches'] += 1
            
            # Display account summary
            console.print(f"\n[bold cyan]ACCOUNT SUMMARY[/bold cyan]")
            summary_table = Table(
                show_header=True,
                header_style="bold magenta",
                border_style="bright_green",
                show_lines=True
            )
            
            summary_table.add_column("Account ID", style="cyan", width=12)
            summary_table.add_column("Unique Roles", style="green", justify="center")
            summary_table.add_column("Unique Policies", style="yellow", justify="center")
            summary_table.add_column("Total Matches", style="red", justify="center")
            
            for account_id in sorted(account_summary.keys()):
                data = account_summary[account_id]
                summary_table.add_row(
                    account_id,
                    str(len(data['roles'])),
                    str(len(data['policies'])),
                    str(data['total_matches'])
                )
            
            console.print(summary_table)
            
        else:
            # No results found
            no_results_table = Table(
                show_header=False,
                border_style="yellow",
                title="[bold yellow]SCAN RESULTS[/bold yellow]",
                title_style="bold yellow"
            )
            no_results_table.add_column("Message", style="yellow", justify="center")
            no_results_table.add_row("No IAM roles found with customer-managed policies containing 'delegated'")
            console.print(no_results_table)
    
    def display_summary(self):
        """Display comprehensive scan summary"""
        # Display detailed results table
        self.display_rich_table()
        
        # Statistics and error summary
        console.print(f"\n[bold cyan]SCAN STATISTICS[/bold cyan]")
        stats_table = Table(show_header=False, border_style="bright_cyan")
        stats_table.add_column("Metric", style="cyan", width=25)
        stats_table.add_column("Value", style="green", justify="right")
        
        stats_table.add_row("Total Matches Found", str(len(self.results)))
        stats_table.add_row("Unique Accounts with Matches", str(len(set(result[0] for result in self.results))))
        stats_table.add_row("Unique Roles Found", str(len(set(f"{result[0]}:{result[1]}" for result in self.results))))
        stats_table.add_row("Unique Policy Names", str(len(set(result[2] for result in self.results))))
        stats_table.add_row("Failed Account Scans", str(len(self.failed_accounts)))
        
        console.print(stats_table)
        
        # Failed accounts details
        if self.failed_accounts:
            console.print(f"\n[bold red]FAILED ACCOUNT SCANS[/bold red]")
            failed_table = Table(
                show_header=True,
                header_style="bold red",
                border_style="red"
            )
            failed_table.add_column("Account ID", style="red")
            failed_table.add_column("Account Name", style="yellow")
            failed_table.add_column("Error", style="white")
            
            for account_id, account_name, error in self.failed_accounts:
                failed_table.add_row(account_id, account_name, error[:80] + "..." if len(error) > 80 else error)
            
            console.print(failed_table)
    
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