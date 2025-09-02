#date: 2025-09-02T16:50:16Z
#url: https://api.github.com/gists/f24fee05bc215d328d551ba35f82e17e
#owner: https://api.github.com/users/RajChowdhury240

#!/usr/bin/env python3
"""
AWS Role Scanner
Scans multiple AWS accounts for roles by name with support for
wildcard, prefix, postfix, exact, and contains matching.

Default match mode: wildcard (*pattern*).
"""

import argparse
import boto3
import fnmatch
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Data Classes
# ----------------------------------------------------------------------
@dataclass
class AccountInfo:
    id: str
    name: str
    ou_path: str

@dataclass
class ScanResult:
    account_name: str
    account_id: str
    role_name: str
    role_path: str
    status: str
    last_modified: Optional[str]
    ou_path: str
    is_avm: bool

# ----------------------------------------------------------------------
# Scanner Class
# ----------------------------------------------------------------------
class AWSRoleScanner:
    def __init__(self, org_session: boto3.Session, assume_role_name: str):
        self.org_session = org_session
        self.assume_role_name = assume_role_name
        self.org_client = org_session.client("organizations")

    def list_accounts(self) -> List[AccountInfo]:
        """List AWS accounts in the org."""
        accounts = []
        paginator = self.org_client.get_paginator("list_accounts")
        for page in paginator.paginate():
            for acct in page["Accounts"]:
                accounts.append(AccountInfo(
                    id=acct["Id"],
                    name=acct["Name"],
                    ou_path="/"  # Simplified
                ))
        return accounts

    def check_avm_tag(self, account_id: str) -> bool:
        """Check if account has AVM tag."""
        try:
            tags = self.org_client.list_tags_for_resource(ResourceId=account_id)
            for tag in tags.get("Tags", []):
                if tag["Key"].lower() == "avm" and tag["Value"].lower() == "true":
                    return True
        except Exception:
            pass
        return False

    def assume_role(self, account_id: str) -> Optional[boto3.Session]:
        """Assume role into account."""
        sts = self.org_session.client("sts")
        role_arn = f"arn:aws:iam::{account_id}:role/{self.assume_role_name}"
        try:
            resp = sts.assume_role(
                RoleArn=role_arn,
                RoleSessionName="RoleScannerSession"
            )
            creds = resp["Credentials"]
            return boto3.Session(
                aws_access_key_id= "**********"
                aws_secret_access_key= "**********"
                aws_session_token= "**********"
            )
        except Exception as e:
            logger.warning(f"Cannot assume role in {account_id}: {e}")
            return None

    def scan_role_in_account(
        self, session: boto3.Session, target_role_name: str, match_mode: str = "wildcard"
    ) -> List[Tuple[str, str, str, Optional[datetime]]]:
        """Scan account for roles with matching logic."""
        iam = session.client("iam")
        matches = []

        try:
            paginator = iam.get_paginator("list_roles")
            for page in paginator.paginate():
                for role in page["Roles"]:
                    name = role["RoleName"]

                    if match_mode == "exact" and name == target_role_name:
                        matches.append((name, role["Path"], "EXISTS", role.get("CreateDate")))
                    elif match_mode == "wildcard" and fnmatch.fnmatch(name, target_role_name):
                        matches.append((name, role["Path"], "EXISTS", role.get("CreateDate")))
                    elif match_mode == "prefix" and name.startswith(target_role_name):
                        matches.append((name, role["Path"], "EXISTS", role.get("CreateDate")))
                    elif match_mode == "postfix" and name.endswith(target_role_name):
                        matches.append((name, role["Path"], "EXISTS", role.get("CreateDate")))
                    elif match_mode == "contains" and target_role_name in name:
                        matches.append((name, role["Path"], "EXISTS", role.get("CreateDate")))

            if not matches:
                matches.append((target_role_name, "/", "MISSING", None))

        except Exception as e:
            logger.error(f"Error scanning roles: {e}")
            matches.append((target_role_name, "/", "ERROR", None))

        return matches

    def scan_single_account(
        self, account: AccountInfo, target_role_name: str, match_mode: str
    ) -> List[ScanResult]:
        """Scan one account for roles."""
        is_avm = self.check_avm_tag(account.id)
        session = self.assume_role(account.id)

        if session is None:
            return [ScanResult(
                account_name=account.name,
                account_id=account.id,
                role_name=target_role_name,
                role_path="/",
                status="ACCESS_DENIED",
                last_modified=None,
                ou_path=account.ou_path,
                is_avm=is_avm
            )]

        results = []
        for role_name, role_path, status, last_modified in self.scan_role_in_account(
            session, target_role_name, match_mode
        ):
            results.append(ScanResult(
                account_name=account.name,
                account_id=account.id,
                role_name=role_name,
                role_path=role_path,
                status=status,
                last_modified=last_modified.strftime("%Y-%m-%d %H:%M:%S") if last_modified else None,
                ou_path=account.ou_path,
                is_avm=is_avm
            ))
        return results

    def scan_accounts(
        self, target_role_name: str, match_mode: str = "wildcard"
    ) -> List[ScanResult]:
        """Scan all accounts concurrently."""
        accounts = self.list_accounts()
        results = []
        console = Console()

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            TextColumn("[green]{task.fields[current_account]}"),
            console=console,
        ) as progress:
            task = progress.add_task(
                "Scanning accounts",
                total=len(accounts),
                current_account=""
            )

            with ThreadPoolExecutor(max_workers=10) as executor:
                future_to_acct = {
                    executor.submit(self.scan_single_account, acct, target_role_name, match_mode): acct
                    for acct in accounts
                }
                for future in as_completed(future_to_acct):
                    acct = future_to_acct[future]
                    try:
                        acct_results = future.result()
                        results.extend(acct_results)
                        progress.update(
                            task, advance=1, current_account=f"{acct.name} ({acct.id})"
                        )
                    except Exception as e:
                        logger.error(f"Error scanning {acct.id}: {e}")
                        results.append(ScanResult(
                            account_name=acct.name,
                            account_id=acct.id,
                            role_name=target_role_name,
                            role_path="/",
                            status="ERROR",
                            last_modified=None,
                            ou_path=acct.ou_path,
                            is_avm=self.check_avm_tag(acct.id)
                        ))
                        progress.advance(task)

        return results

# ----------------------------------------------------------------------
# Output Functions
# ----------------------------------------------------------------------
def save_csv(results: List[ScanResult], filename: str):
    import csv
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=asdict(results[0]).keys())
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))

def save_html(results: List[ScanResult], filename: str):
    from jinja2 import Template
    template = Template("""
    <html>
    <head><title>AWS Role Scan</title></head>
    <body>
      <h1>AWS Role Scan Results</h1>
      <table border="1" cellpadding="5" cellspacing="0">
        <tr>
        {% for key in results[0].keys() %}
          <th>{{ key }}</th>
        {% endfor %}
        </tr>
        {% for row in results %}
        <tr>
          {% for val in row.values() %}
          <td>{{ val }}</td>
          {% endfor %}
        </tr>
        {% endfor %}
      </table>
    </body>
    </html>
    """)
    with open(filename, "w") as f:
        f.write(template.render(results=[asdict(r) for r in results]))

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="AWS Role Scanner")
    parser.add_argument("--role", required=True, help="Role name or pattern to search")
    parser.add_argument("--assume-role", required=True, help="Role name to assume into accounts")
    parser.add_argument("--match-mode",
                        choices=["exact", "wildcard", "prefix", "postfix", "contains"],
                        default="wildcard",
                        help="Role name match mode (default: wildcard)")
    parser.add_argument("--csv", default="results.csv", help="CSV output file")
    parser.add_argument("--html", default="results.html", help="HTML output file")

    args = parser.parse_args()

    org_session = boto3.Session()
    scanner = AWSRoleScanner(org_session, args.assume_role)
    results = scanner.scan_accounts(args.role, match_mode=args.match_mode)

    if results:
        save_csv(results, args.csv)
        save_html(results, args.html)
        print(f"✅ Results saved to {args.csv} and {args.html}")
    else:
        print("⚠️ No results found")

if __name__ == "__main__":
    main()__ == "__main__":
    main()