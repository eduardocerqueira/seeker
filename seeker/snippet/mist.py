#date: 2025-07-07T17:08:16Z
#url: https://api.github.com/gists/a4b2cfd658d19c1261f3eb05e06b7d4a
#owner: https://api.github.com/users/RajChowdhury240

#!/usr/bin/env python3
"""
AWS IAM Role Audit Script
Scans all active AWS accounts for roles with specific prefixes and analyzes their permissions
"""

import boto3
import json
import csv
import concurrent.futures
import logging
from datetime import datetime
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass, asdict
from botocore.exceptions import ClientError, NoCredentialsError
import html

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('aws_iam_audit.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class RoleInfo:
    """Data class to store role information"""
    account_id: str
    account_name: str
    role_name: str
    role_arn: str
    creation_date: str
    last_used: str
    trust_relationship: str
    attached_policies: List[str]
    inline_policies: List[str]
    permission_level: str
    has_high_permissions: bool
    trust_entities: List[str]

class AWSIAMRoleAuditor:
    """Main class for AWS IAM Role auditing"""
    
    def __init__(self):
        self.organizations_client = None
        self.excluded_ous = ['decom', 'sandbox', 'innovate']
        self.target_role = 'ca-iam-cie-engineer'
        self.target_prefixes = ['ta-', 'ca-']
        self.excluded_prefixes = ['ca-svc-']
        self.excluded_keywords = ['config-audit']
        self.read_only_actions = {
            'list', 'describe', 'get', 'read', 'view', 'select', 'query',
            'scan', 'count', 'head', 'exists', 'check', 'validate'
        }
        
    def initialize_clients(self) -> bool:
        """Initialize AWS clients"""
        try:
            self.organizations_client = boto3.client('organizations')
            logger.info("AWS clients initialized successfully")
            return True
        except NoCredentialsError:
            logger.error("AWS credentials not found. Please configure your credentials.")
            return False
        except Exception as e:
            logger.error(f"Error initializing AWS clients: {str(e)}")
            return False
    
    def get_all_accounts(self) -> List[Dict]:
        """Get all active AWS accounts excluding specified OUs"""
        try:
            logger.info("Fetching all AWS accounts...")
            
            # Get all accounts
            paginator = self.organizations_client.get_paginator('list_accounts')
            all_accounts = []
            
            for page in paginator.paginate():
                all_accounts.extend(page['Accounts'])
            
            # Filter active accounts
            active_accounts = [
                acc for acc in all_accounts 
                if acc['Status'] == 'ACTIVE'
            ]
            
            # Get organizational units for each account
            filtered_accounts = []
            for account in active_accounts:
                try:
                    # Get the organizational units for this account
                    response = self.organizations_client.list_parents(
                        ChildId=account['Id']
                    )
                    
                    # Check if account is in excluded OUs
                    is_excluded = False
                    for parent in response['Parents']:
                        if parent['Type'] == 'ORGANIZATIONAL_UNIT':
                            try:
                                ou_info = self.organizations_client.describe_organizational_unit(
                                    OrganizationalUnitId=parent['Id']
                                )
                                ou_name = ou_info['OrganizationalUnit']['Name'].lower()
                                
                                if any(excluded_ou.lower() in ou_name for excluded_ou in self.excluded_ous):
                                    is_excluded = True
                                    logger.info(f"Excluding account {account['Id']} from OU: {ou_name}")
                                    break
                            except ClientError as e:
                                logger.warning(f"Could not get OU info for {parent['Id']}: {str(e)}")
                    
                    if not is_excluded:
                        filtered_accounts.append(account)
                        
                except ClientError as e:
                    logger.warning(f"Could not get parent info for account {account['Id']}: {str(e)}")
                    # Include account if we can't determine its OU
                    filtered_accounts.append(account)
            
            logger.info(f"Found {len(filtered_accounts)} active accounts (excluding specified OUs)")
            return filtered_accounts
            
        except ClientError as e:
            logger.error(f"Error fetching accounts: {str(e)}")
            return []
    
    def assume_role_in_account(self, account_id: str, role_name: str) -> Optional[boto3.Session]:
        """Assume a role in the specified account"""
        try:
            sts_client = boto3.client('sts')
            role_arn = f"arn:aws:iam::{account_id}:role/{role_name}"
            
            response = sts_client.assume_role(
                RoleArn=role_arn,
                RoleSessionName=f"audit-session-{account_id}"
            )
            
            credentials = response['Credentials']
            session = boto3.Session(
                aws_access_key_id= "**********"
                aws_secret_access_key= "**********"
                aws_session_token= "**********"
            )
            
            return session
            
        except ClientError as e:
            logger.warning(f"Could not assume role {role_name} in account {account_id}: {str(e)}")
            return None
    
    def is_read_only_permission(self, action: str) -> bool:
        """Check if an action is read-only"""
        action_lower = action.lower()
        
        # Check if action starts with read-only verbs
        for read_action in self.read_only_actions:
            if action_lower.startswith(read_action):
                return True
        
        # Check for specific read-only patterns
        read_only_patterns = [
            ':list*', ':describe*', ':get*', ':read*', ':view*',
            ':select*', ':query*', ':scan*', ':count*', ':head*'
        ]
        
        for pattern in read_only_patterns:
            if pattern.replace('*', '') in action_lower:
                return True
        
        return False
    
    def analyze_policy_permissions(self, policy_document: str) -> Tuple[str, bool]:
        """Analyze policy document to determine permission level"""
        try:
            policy = json.loads(policy_document)
            statements = policy.get('Statement', [])
            
            if not isinstance(statements, list):
                statements = [statements]
            
            has_high_permissions = False
            all_actions = []
            
            for statement in statements:
                if statement.get('Effect') == 'Allow':
                    actions = statement.get('Action', [])
                    if isinstance(actions, str):
                        actions = [actions]
                    
                    for action in actions:
                        all_actions.append(action)
                        
                        # Check for admin permissions
                        if action == '*' or action.endswith(':*'):
                            has_high_permissions = True
                        
                        # Check for specific high-risk actions
                        high_risk_actions = [
                            'create', 'delete', 'update', 'modify', 'put', 'post',
                            'attach', 'detach', 'associate', 'disassociate', 'add',
                            'remove', 'replace', 'change', 'set', 'enable', 'disable',
                            'start', 'stop', 'terminate', 'launch', 'invoke', 'execute'
                        ]
                        
                        if not self.is_read_only_permission(action):
                            for high_risk in high_risk_actions:
                                if high_risk in action.lower():
                                    has_high_permissions = True
                                    break
            
            # Determine permission level
            if has_high_permissions:
                permission_level = "High (Write/Admin)"
            else:
                permission_level = "Read Only"
            
            return permission_level, has_high_permissions
            
        except json.JSONDecodeError:
            logger.warning("Could not parse policy document")
            return "Unknown", True  # Assume high permissions if we can't parse
        except Exception as e:
            logger.warning(f"Error analyzing policy: {str(e)}")
            return "Unknown", True
    
    def get_trust_relationship_entities(self, trust_policy: str) -> List[str]:
        """Extract entities that can assume the role from trust policy"""
        try:
            policy = json.loads(trust_policy)
            statements = policy.get('Statement', [])
            
            if not isinstance(statements, list):
                statements = [statements]
            
            trust_entities = []
            
            for statement in statements:
                if statement.get('Effect') == 'Allow':
                    principals = statement.get('Principal', {})
                    
                    if isinstance(principals, str):
                        trust_entities.append(principals)
                    elif isinstance(principals, dict):
                        for principal_type, principal_values in principals.items():
                            if isinstance(principal_values, str):
                                trust_entities.append(f"{principal_type}: {principal_values}")
                            elif isinstance(principal_values, list):
                                for value in principal_values:
                                    trust_entities.append(f"{principal_type}: {value}")
            
            return trust_entities
            
        except json.JSONDecodeError:
            logger.warning("Could not parse trust policy")
            return ["Unknown"]
        except Exception as e:
            logger.warning(f"Error parsing trust policy: {str(e)}")
            return ["Unknown"]
    
    def scan_roles_in_account(self, account: Dict) -> List[RoleInfo]:
        """Scan roles in a specific account"""
        account_id = account['Id']
        account_name = account.get('Name', 'Unknown')
        
        logger.info(f"Scanning roles in account {account_id} ({account_name})")
        
        # Assume role in the account
        session = self.assume_role_in_account(account_id, self.target_role)
        if not session:
            logger.warning(f"Skipping account {account_id} - could not assume role")
            return []
        
        try:
            iam_client = session.client('iam')
            roles_info = []
            
            # Get all roles
            paginator = iam_client.get_paginator('list_roles')
            
            for page in paginator.paginate():
                for role in page['Roles']:
                    role_name = role['RoleName']
                    
                    # Check if role matches our criteria
                    if not any(role_name.startswith(prefix) for prefix in self.target_prefixes):
                        continue
                    
                    # Skip excluded prefixes
                    if any(role_name.startswith(prefix) for prefix in self.excluded_prefixes):
                        continue
                    
                    # Skip excluded keywords
                    if any(keyword in role_name.lower() for keyword in self.excluded_keywords):
                        continue
                    
                    # Get role details
                    try:
                        role_details = iam_client.get_role(RoleName=role_name)
                        role_data = role_details['Role']
                        
                        # Get attached policies
                        attached_policies_response = iam_client.list_attached_role_policies(
                            RoleName=role_name
                        )
                        attached_policies = [
                            policy['PolicyName'] 
                            for policy in attached_policies_response['AttachedPolicies']
                        ]
                        
                        # Get inline policies
                        inline_policies_response = iam_client.list_role_policies(
                            RoleName=role_name
                        )
                        inline_policies = inline_policies_response['PolicyNames']
                        
                        # Analyze permissions
                        has_high_permissions = False
                        permission_level = "Read Only"
                        
                        # Check attached policies
                        for policy_name in attached_policies:
                            try:
                                policy_arn = None
                                for policy in attached_policies_response['AttachedPolicies']:
                                    if policy['PolicyName'] == policy_name:
                                        policy_arn = policy['PolicyArn']
                                        break
                                
                                if policy_arn:
                                    policy_response = iam_client.get_policy(PolicyArn=policy_arn)
                                    policy_version = iam_client.get_policy_version(
                                        PolicyArn=policy_arn,
                                        VersionId=policy_response['Policy']['DefaultVersionId']
                                    )
                                    
                                    policy_doc = json.dumps(policy_version['PolicyVersion']['Document'])
                                    perm_level, has_high_perms = self.analyze_policy_permissions(policy_doc)
                                    
                                    if has_high_perms:
                                        has_high_permissions = True
                                        permission_level = perm_level
                                        break
                            except ClientError as e:
                                logger.warning(f"Could not analyze policy {policy_name}: {str(e)}")
                        
                        # Check inline policies
                        for policy_name in inline_policies:
                            try:
                                policy_response = iam_client.get_role_policy(
                                    RoleName=role_name,
                                    PolicyName=policy_name
                                )
                                
                                policy_doc = json.dumps(policy_response['PolicyDocument'])
                                perm_level, has_high_perms = self.analyze_policy_permissions(policy_doc)
                                
                                if has_high_perms:
                                    has_high_permissions = True
                                    permission_level = perm_level
                                    break
                            except ClientError as e:
                                logger.warning(f"Could not analyze inline policy {policy_name}: {str(e)}")
                        
                        # Skip if role has only read-only permissions
                        if not has_high_permissions:
                            logger.debug(f"Skipping role {role_name} - read-only permissions")
                            continue
                        
                        # Get trust relationship
                        trust_policy = json.dumps(role_data['AssumeRolePolicyDocument'])
                        trust_entities = self.get_trust_relationship_entities(trust_policy)
                        
                        # Create role info
                        role_info = RoleInfo(
                            account_id=account_id,
                            account_name=account_name,
                            role_name=role_name,
                            role_arn=role_data['Arn'],
                            creation_date=role_data['CreateDate'].isoformat(),
                            last_used=role_data.get('RoleLastUsed', {}).get('LastUsedDate', 'Never').isoformat() if isinstance(role_data.get('RoleLastUsed', {}).get('LastUsedDate'), datetime) else str(role_data.get('RoleLastUsed', {}).get('LastUsedDate', 'Never')),
                            trust_relationship=trust_policy,
                            attached_policies=attached_policies,
                            inline_policies=inline_policies,
                            permission_level=permission_level,
                            has_high_permissions=has_high_permissions,
                            trust_entities=trust_entities
                        )
                        
                        roles_info.append(role_info)
                        logger.info(f"Found role: {role_name} with {permission_level} permissions")
                        
                    except ClientError as e:
                        logger.warning(f"Could not get details for role {role_name}: {str(e)}")
                        continue
            
            logger.info(f"Found {len(roles_info)} roles with high permissions in account {account_id}")
            return roles_info
            
        except ClientError as e:
            logger.error(f"Error scanning roles in account {account_id}: {str(e)}")
            return []
    
    def scan_all_accounts(self, accounts: List[Dict]) -> List[RoleInfo]:
        """Scan all accounts using multithreading"""
        logger.info(f"Starting scan of {len(accounts)} accounts using multithreading")
        
        all_roles = []
        
        # Use ThreadPoolExecutor for concurrent scanning
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            # Submit all tasks
            future_to_account = {
                executor.submit(self.scan_roles_in_account, account): account
                for account in accounts
            }
            
            # Collect results
            for future in concurrent.futures.as_completed(future_to_account):
                account = future_to_account[future]
                try:
                    roles = future.result()
                    all_roles.extend(roles)
                except Exception as e:
                    logger.error(f"Error scanning account {account['Id']}: {str(e)}")
        
        logger.info(f"Total roles found across all accounts: {len(all_roles)}")
        return all_roles
    
    def generate_csv_report(self, roles: List[RoleInfo], filename: str = None) -> str:
        """Generate CSV report"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"aws_iam_roles_audit_{timestamp}.csv"
        
        logger.info(f"Generating CSV report: {filename}")
        
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'Account ID', 'Account Name', 'Role Name', 'Role ARN',
                'Creation Date', 'Last Used', 'Permission Level',
                'Attached Policies', 'Inline Policies', 'Trust Entities'
            ]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for role in roles:
                writer.writerow({
                    'Account ID': role.account_id,
                    'Account Name': role.account_name,
                    'Role Name': role.role_name,
                    'Role ARN': role.role_arn,
                    'Creation Date': role.creation_date,
                    'Last Used': role.last_used,
                    'Permission Level': role.permission_level,
                    'Attached Policies': ', '.join(role.attached_policies),
                    'Inline Policies': ', '.join(role.inline_policies),
                    'Trust Entities': ', '.join(role.trust_entities)
                })
        
        logger.info(f"CSV report generated: {filename}")
        return filename
    
    def generate_html_report(self, roles: List[RoleInfo], filename: str = None) -> str:
        """Generate HTML report"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"aws_iam_roles_audit_{timestamp}.html"
        
        logger.info(f"Generating HTML report: {filename}")
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AWS IAM Roles Audit Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; margin-bottom: 20px; }}
                .summary {{ background-color: #e7f3ff; padding: 15px; margin-bottom: 20px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .high-permission {{ background-color: #ffcccc; }}
                .trust-policy {{ max-width: 300px; word-wrap: break-word; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>AWS IAM Roles Audit Report</h1>
                <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            </div>
            
            <div class="summary">
                <h2>Summary</h2>
                <p>Total roles found: {len(roles)}</p>
                <p>High permission roles: {sum(1 for role in roles if role.has_high_permissions)}</p>
                <p>Accounts scanned: {len(set(role.account_id for role in roles))}</p>
            </div>
            
            <table>
                <tr>
                    <th>Account ID</th>
                    <th>Account Name</th>
                    <th>Role Name</th>
                    <th>Permission Level</th>
                    <th>Creation Date</th>
                    <th>Last Used</th>
                    <th>Attached Policies</th>
                    <th>Inline Policies</th>
                    <th>Trust Entities</th>
                </tr>
        """
        
        for role in roles:
            row_class = "high-permission" if role.has_high_permissions else ""
            html_content += f"""
                <tr class="{row_class}">
                    <td>{html.escape(role.account_id)}</td>
                    <td>{html.escape(role.account_name)}</td>
                    <td>{html.escape(role.role_name)}</td>
                    <td>{html.escape(role.permission_level)}</td>
                    <td>{html.escape(role.creation_date)}</td>
                    <td>{html.escape(role.last_used)}</td>
                    <td>{html.escape(', '.join(role.attached_policies))}</td>
                    <td>{html.escape(', '.join(role.inline_policies))}</td>
                    <td class="trust-policy">{html.escape(', '.join(role.trust_entities))}</td>
                </tr>
            """
        
        html_content += """
            </table>
        </body>
        </html>
        """
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML report generated: {filename}")
        return filename
    
    def run_audit(self) -> bool:
        """Run the complete audit process"""
        logger.info("Starting AWS IAM Role Audit")
        
        # Initialize clients
        if not self.initialize_clients():
            return False
        
        # Get all accounts
        accounts = self.get_all_accounts()
        if not accounts:
            logger.error("No accounts found or error fetching accounts")
            return False
        
        # Scan all accounts
        roles = self.scan_all_accounts(accounts)
        if not roles:
            logger.warning("No roles found matching criteria")
            return True
        
        # Generate reports
        csv_file = self.generate_csv_report(roles)
        html_file = self.generate_html_report(roles)
        
        logger.info(f"Audit completed successfully!")
        logger.info(f"CSV Report: {csv_file}")
        logger.info(f"HTML Report: {html_file}")
        logger.info(f"Total roles with high permissions: {len(roles)}")
        
        return True

def main():
    """Main function to run the audit"""
    auditor = AWSIAMRoleAuditor()
    
    try:
        success = auditor.run_audit()
        if success:
            print("Audit completed successfully!")
        else:
            print("Audit failed. Check logs for details.")
    except KeyboardInterrupt:
        print("\nAudit interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        print(f"Unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()r(e)}")

if __name__ == "__main__":
    main()