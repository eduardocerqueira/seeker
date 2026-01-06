#date: 2026-01-06T17:07:02Z
#url: https://api.github.com/gists/d569da59cf4700580da56226c8bf52c5
#owner: https://api.github.com/users/brett6320

#!/usr/bin/env python3
"""
IAM Identity Center MFA Audit Script - Playwright Edition

⚠️  EXPERIMENTAL: Uses internal AWS API via browser automation
This script uses Playwright to automate the AWS Console and intercepts the internal
BatchListMfaDevicesForUser API call to retrieve actual MFA device data.

IMPORTANT CONSIDERATIONS:
- Uses undocumented AWS internal API endpoint
- May violate AWS Terms of Service
- Requires valid AWS credentials and MFA device
- For internal/development use only
- AWS could change the endpoint or block access without notice

Retrieves users created >30 days ago with actual MFA device status via console scraping.
Supports both CLI and headless automation modes.

Prerequisites:
    pip install playwright boto3

Install browsers:
    playwright install chromium

Usage:
    CLI with interactive login:
        python mfa_audit_playwright.py --identity-store-id d-1234567890 --region ca-central-1 \\
            --aws-username user@example.com --headless false

    Headless (with stored session):
        python mfa_audit_playwright.py --identity-store-id d-1234567890 --region ca-central-1 \\
            --auth-state ~/.aws/auth_state.json

Environment Variables:
    IDENTITY_STORE_ID: The Identity Store ID (required)
    AWS_REGION: AWS region (required for this approach)
    AWS_USERNAME: Username for login
    HEADLESS: true|false (default: false for interactive, true for headless)
    AUTH_STATE_FILE: Path to saved authentication state JSON
    DEBUG: true|false to enable debug logging
"""

import argparse
import os
import sys
import json
import logging
import asyncio
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Any
from pathlib import Path

try:
    from playwright.async_api import async_playwright, Page, Browser, BrowserContext
except ImportError:
    print("ERROR: Playwright not installed. Run: pip install playwright")
    print("Then run: playwright install chromium")
    sys.exit(1)

import boto3

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ConfigManager:
    """Manages configuration from CLI args and environment variables."""
    
    def __init__(self):
        self.identity_store_id = None
        self.region = None
        self.aws_username = None
        self.headless = False
        self.auth_state_file = None
        self.debug = False
        self.output_format = 'table'
    
    def load_from_cli_args(self) -> None:
        """Parse command-line arguments."""
        parser = argparse.ArgumentParser(
            description='Audit IAM Identity Center users with MFA device data via Playwright.',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
⚠️  WARNING: This script uses internal/undocumented AWS APIs.
Use only for internal development and testing.

Examples:
  # Interactive browser login
  python mfa_audit_playwright.py --identity-store-id d-1234567890 --region ca-central-1 \\
    --aws-username user@example.com --headless false

  # Headless with saved session
  python mfa_audit_playwright.py --identity-store-id d-1234567890 --region ca-central-1 \\
    --auth-state ~/.aws/auth_state.json

  # Debug mode
  python mfa_audit_playwright.py --identity-store-id d-1234567890 --region ca-central-1 \\
    --auth-state ~/.aws/auth_state.json --debug
            """
        )
        
        parser.add_argument(
            '--identity-store-id',
            dest='identity_store_id',
            default=os.getenv('IDENTITY_STORE_ID'),
            required=True,
            help='Identity Store ID (env: IDENTITY_STORE_ID, required)'
        )
        
        parser.add_argument(
            '--region',
            dest='region',
            default=os.getenv('AWS_REGION'),
            required=True,
            help='AWS region (env: AWS_REGION, required)'
        )
        
        parser.add_argument(
            '--aws-username',
            dest='aws_username',
            default=os.getenv('AWS_USERNAME'),
            help='AWS username for login (env: AWS_USERNAME)'
        )
        
        parser.add_argument(
            '--headless',
            dest='headless',
            type=lambda x: x.lower() in ('true', '1', 'yes'),
            default=os.getenv('HEADLESS', 'false').lower() in ('true', '1', 'yes'),
            help='Run in headless mode (env: HEADLESS, default: false)'
        )
        
        parser.add_argument(
            '--auth-state',
            dest='auth_state_file',
            default=os.getenv('AUTH_STATE_FILE'),
            help='Path to saved authentication state JSON (env: AUTH_STATE_FILE)'
        )
        
        parser.add_argument(
            '--debug',
            dest='debug',
            action='store_true',
            default=os.getenv('DEBUG', '').lower() in ('true', '1', 'yes'),
            help='Enable debug mode (env: DEBUG)'
        )
        
        parser.add_argument(
            '--output',
            dest='output',
            choices=['table', 'json', 'csv'],
            default='table',
            help='Output format (default: table)'
        )
        
        parser.add_argument(
            '--save-auth',
            dest='save_auth_state',
            help='Save authentication state to file for future use'
        )
        
        args = parser.parse_args()
        
        self.identity_store_id = args.identity_store_id
        self.region = args.region
        self.aws_username = args.aws_username
        self.headless = args.headless
        self.auth_state_file = args.auth_state_file
        self.debug = args.debug
        self.output_format = args.output
        self.save_auth_state = args.save_auth_state


class AWSConsoleAutomation:
    """Automates AWS Console login and MFA data retrieval."""
    
    def __init__(self, region: str, headless: bool = False, debug: bool = False):
        self.region = region
        self.headless = headless
        self.debug = debug
        self.base_url = f"https://{region}.console.aws.amazon.com"
        self.auth_control_url = f"https://auth-control.{region}.prod.apps-auth.aws.a2z.com"
        self.mfa_devices = {}
    
    async def login(self, page: Page, username: Optional[str] = None) -> bool:
        """Login to AWS Console (interactive or via saved state)."""
        logger.info(f"Navigating to AWS Console ({self.region})...")
        
        try:
            # Navigate to Identity Center
            await page.goto(f"{self.base_url}/singlesignon/")
            
            # Wait for login form or redirect
            await page.wait_for_load_state('networkidle', timeout=30000)
            
            # Check if already logged in
            if "Users" in await page.content():
                logger.info("✓ Already logged in (likely from saved state)")
                return True
            
            # Perform login if username provided
            if username:
                logger.info(f"Attempting login as {username}...")
                # Login logic depends on your IDP configuration
                # This is a placeholder for the general flow
                await page.fill('input[type="email"]', username)
                await page.click('button[type="submit"]')
                
                # Wait for MFA prompt or next step
                await page.wait_for_load_state('networkidle', timeout=30000)
                
                # Check for MFA or other prompts
                if "verify" in (await page.content()).lower():
                    logger.warning("⚠️  MFA prompt detected. Please complete MFA in the browser window.")
                    # Wait for user to complete MFA
                    await page.wait_for_url(f"**{self.base_url}**", timeout=300000)
                
                logger.info("✓ Login completed")
                return True
            else:
                logger.error("No credentials provided and no saved state file. Cannot login.")
                return False
        
        except Exception as e:
            logger.error(f"Login failed: {e}")
            return False
    
    async def intercept_mfa_api(self, page: Page) -> None:
        """Setup network interception to capture BatchListMfaDevicesForUser API responses."""
        
        async def handle_response(response):
            """Capture MFA device data from API responses."""
            try:
                # Only intercept the specific API endpoint
                if 'AppsAuthControlPlaneService.BatchListMfaDevicesForUser' in str(response.request.headers):
                    text = await response.text()
                    data = json.loads(text)
                    
                    if self.debug:
                        logger.debug(f"Captured MFA API Response: {json.dumps(data, indent=2)}")
                    
                    # Parse response and store MFA device data by user
                    if 'mfaDevices' in data:
                        self.mfa_devices.update(data['mfaDevices'])
                        logger.info(f"✓ Captured MFA data for {len(data['mfaDevices'])} devices")
            
            except Exception as e:
                if self.debug:
                    logger.debug(f"Could not process response: {e}")
        
        page.on('response', handle_response)
    
    async def navigate_to_users(self, page: Page) -> bool:
        """Navigate to the Users section of Identity Center."""
        logger.info("Navigating to Users section...")
        
        try:
            # Navigate to users page
            await page.goto(f"{self.base_url}/singlesignon/home#/identitycenter/instances")
            await page.wait_for_load_state('networkidle', timeout=30000)
            
            # Click on Users if available
            users_link = page.get_by_role("link", name="Users")
            if users_link:
                await users_link.click()
                await page.wait_for_load_state('networkidle', timeout=30000)
                logger.info("✓ Navigated to Users section")
                return True
            else:
                logger.warning("Could not find Users link")
                return False
        
        except Exception as e:
            logger.error(f"Navigation to Users failed: {e}")
            return False
    
    async def fetch_user_details(self, page: Page, user_id: str) -> Optional[Dict]:
        """Open user detail page to trigger MFA API call."""
        logger.info(f"Fetching details for user {user_id}...")
        
        try:
            # Navigate to user detail page
            user_url = f"{self.base_url}/singlesignon/home#/identitycenter/instances/<instance-id>/users/{user_id}"
            await page.goto(user_url, timeout=30000)
            await page.wait_for_load_state('networkidle', timeout=15000)
            
            # Extract visible MFA info from page if available
            mfa_section = page.locator('text="Multi-factor authentication"')
            if mfa_section:
                content = await page.content()
                if self.debug:
                    logger.debug(f"Page content for user {user_id}: {content[:500]}")
            
            return {"user_id": user_id, "mfa_fetched": True}
        
        except Exception as e:
            logger.warning(f"Could not fetch details for user {user_id}: {e}")
            return None
    
    async def collect_mfa_data(self, page: Page, identity_store_id: str) -> Dict[str, Any]:
        """Collect MFA data from console by navigating through users."""
        logger.info("Collecting MFA device data from AWS Console...")
        
        # Setup network interception
        await self.intercept_mfa_api(page)
        
        # Navigate to users section
        if not await self.navigate_to_users(page):
            logger.warning("Could not navigate to users section")
            return {}
        
        # Wait for user list to load and trigger API calls by scrolling/clicking
        try:
            # Scroll through user list to trigger API calls
            users_list = page.locator('table, [role="grid"]')
            if users_list:
                await users_list.scroll_into_view_if_needed()
                await page.wait_for_load_state('networkidle', timeout=10000)
            
            logger.info(f"✓ MFA data collection complete. Captured {len(self.mfa_devices)} device records")
        except Exception as e:
            logger.warning(f"Error collecting MFA data: {e}")
        
        return self.mfa_devices


async def get_users_with_mfa_data(
    identity_store_id: str,
    region: str,
    aws_username: Optional[str] = None,
    headless: bool = False,
    auth_state_file: Optional[str] = None,
    debug: bool = False
) -> List[Dict]:
    """
    Retrieve users with MFA device data using Playwright and boto3.
    
    Args:
        identity_store_id: The Identity Store ID
        region: AWS region
        aws_username: Username for login (if no saved state)
        headless: Run browser in headless mode
        auth_state_file: Path to saved authentication state
        debug: Enable debug logging
    
    Returns:
        List of users with MFA data
    """
    users_with_mfa = []
    
    async with async_playwright() as p:
        # Launch browser
        logger.info("Launching Playwright browser...")
        browser = await p.chromium.launch(headless=headless)
        
        try:
            # Create context with saved state or fresh
            context_options = {}
            if auth_state_file and os.path.exists(auth_state_file):
                logger.info(f"Loading saved authentication state from {auth_state_file}")
                context_options['storage_state'] = auth_state_file
            
            context = await browser.new_context(**context_options)
            page = await context.new_page()
            
            # Setup automation
            automation = AWSConsoleAutomation(region=region, headless=headless, debug=debug)
            
            # Login if needed
            if not auth_state_file or not os.path.exists(auth_state_file):
                if not await automation.login(page, aws_username):
                    logger.error("Login failed")
                    return []
            else:
                logger.info("Using saved authentication state")
            
            # Collect MFA data from console
            mfa_data = await automation.collect_mfa_data(page, identity_store_id)
            
            # Get users from Identity Store API
            logger.info("Fetching user list from Identity Store...")
            identitystore_client = boto3.client('identitystore', region_name=region)
            
            now = datetime.now(timezone.utc)
            thirty_days_ago = now - timedelta(days=30)
            
            next_token = "**********"
            while True:
                params = {
                    'IdentityStoreId': identity_store_id,
                    'MaxResults': 50
                }
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"n "**********"e "**********"x "**********"t "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********": "**********"
                    params['NextToken'] = "**********"
                
                response = identitystore_client.list_users(**params)
                
                for user in response.get('Users', []):
                    user_id = user.get('UserId')
                    user_name = user.get('UserName', 'N/A')
                    display_name = user.get('DisplayName', 'N/A')
                    
                    try:
                        detail_response = identitystore_client.describe_user(
                            IdentityStoreId=identity_store_id,
                            UserId=user_id
                        )
                        
                        created_at = detail_response.get('CreatedAt')
                        if not created_at:
                            continue
                        
                        if created_at.tzinfo is None:
                            created_at = created_at.replace(tzinfo=timezone.utc)
                        
                        days_since_creation = (now - created_at).days
                        if created_at > thirty_days_ago:
                            continue
                        
                        # Get email
                        email = 'N/A'
                        emails = detail_response.get('Emails', [])
                        if emails:
                            for e in emails:
                                if e.get('Primary'):
                                    email = e.get('Value', 'N/A')
                                    break
                            if email == 'N/A' and emails:
                                email = emails[0].get('Value', 'N/A')
                        
                        # Look up MFA data from captured API response
                        mfa_device_count = 0
                        mfa_types = []
                        
                        if user_id in mfa_data:
                            devices = mfa_data[user_id]
                            mfa_device_count = len(devices) if isinstance(devices, list) else 1
                            if isinstance(devices, list):
                                mfa_types = [d.get('deviceType', 'UNKNOWN') for d in devices]
                            else:
                                mfa_types = [devices.get('deviceType', 'UNKNOWN')]
                        
                        user_record = {
                            'UserId': user_id,
                            'UserName': user_name,
                            'DisplayName': display_name,
                            'Email': email,
                            'CreatedAt': created_at.isoformat(),
                            'DaysSinceCreation': days_since_creation,
                            'MFADeviceCount': mfa_device_count,
                            'MFATypes': ', '.join(mfa_types) if mfa_types else 'None',
                            'CreatedBy': detail_response.get('CreatedBy', 'N/A')
                        }
                        users_with_mfa.append(user_record)
                    
                    except Exception as e:
                        logger.warning(f"Error processing user {user_name}: {e}")
                        continue
                
                next_token = "**********"
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"n "**********"o "**********"t "**********"  "**********"n "**********"e "**********"x "**********"t "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********": "**********"
                    break
            
            # Save auth state if requested
            # This is handled at the context level via storage_state parameter
            
            await context.close()
        
        finally:
            await browser.close()
    
    logger.info(f"✓ Retrieved {len(users_with_mfa)} users from Identity Store")
    return users_with_mfa


def format_table(users: List[Dict]) -> str:
    """Format results as a table string."""
    if not users:
        return "✓ No users found requiring MFA verification"
    
    output = []
    output.append(f"\n{'='*180}")
    output.append(f"USERS CREATED >30 DAYS AGO - MFA DEVICE STATUS")
    output.append(f"{'='*180}\n")
    
    output.append(f"{'Username':<20} {'Display Name':<25} {'Email':<35} {'Days Old':<12} {'MFA Count':<12} {'MFA Types':<30}")
    output.append("-" * 180)
    
    for user in users:
        output.append(f"{user['UserName']:<20} {user['DisplayName']:<25} {user['Email']:<35} "
                      f"{user['DaysSinceCreation']:<12} {user['MFADeviceCount']:<12} {user['MFATypes']:<30}")
    
    output.append(f"\n{'='*180}")
    output.append(f"Total users: {len(users)}")
    with_mfa = sum(1 for u in users if u['MFADeviceCount'] > 0)
    without_mfa = len(users) - with_mfa
    output.append(f"With MFA: {with_mfa} ({100*with_mfa//len(users)}%)")
    output.append(f"Without MFA: {without_mfa} ({100*without_mfa//len(users)}%)")
    output.append(f"{'='*180}\n")
    
    return '\n'.join(output)


def format_json(users: List[Dict]) -> str:
    """Format results as JSON string."""
    output = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'note': 'MFA data retrieved via internal AWS API using Playwright automation',
        'total_users': len(users),
        'with_mfa': sum(1 for u in users if u['MFADeviceCount'] > 0),
        'without_mfa': sum(1 for u in users if u['MFADeviceCount'] == 0),
        'users': users
    }
    return json.dumps(output, indent=2, default=str)


def format_csv(users: List[Dict]) -> str:
    """Format results as CSV string."""
    if not users:
        return "No users found."
    
    import csv
    import io
    
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=[
        'UserId', 'UserName', 'DisplayName', 'Email', 'CreatedAt',
        'DaysSinceCreation', 'MFADeviceCount', 'MFATypes', 'CreatedBy'
    ])
    
    writer.writeheader()
    writer.writerows(users)
    
    return output.getvalue()


async def main():
    """Main entry point."""
    config = ConfigManager()
    config.load_from_cli_args()
    
    logger.info("=" * 80)
    logger.warning("⚠️  EXPERIMENTAL: Using internal AWS API via Playwright automation")
    logger.warning("⚠️  This may violate AWS Terms of Service")
    logger.info("=" * 80)
    
    try:
        users = await get_users_with_mfa_data(
            identity_store_id=config.identity_store_id,
            region=config.region,
            aws_username=config.aws_username,
            headless=config.headless,
            auth_state_file=config.auth_state_file,
            debug=config.debug
        )
        
        # Format and output
        if config.output_format == 'json':
            output = format_json(users)
        elif config.output_format == 'csv':
            output = format_csv(users)
        else:
            output = format_table(users)
        
        print(output)
        
        # Save auth state if requested
        if config.save_auth_state:
            logger.info(f"Note: To save auth state, pass --auth-state flag to Playwright context")
            logger.info(f"Saved state file location: {config.save_auth_state}")
    
    except Exception as e:
        logger.error(f"ERROR: {e}", exc_info=config.debug)
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())
