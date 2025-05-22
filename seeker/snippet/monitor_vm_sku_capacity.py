#date: 2025-05-22T17:03:54Z
#url: https://api.github.com/gists/7f8fd1c3408464e5ea652301017c701c
#owner: https://api.github.com/users/ricmmartins

#!/usr/bin/env python
"""
Azure VM SKU Capacity Monitor

This script checks the availability of specific VM SKUs in Azure regions
and provides information about capacity constraints and alternative options.
"""

import argparse
import datetime
import json
import logging
import os
import re
import subprocess
import sys
from typing import Dict, List, Any, Optional, Tuple

from azure.identity import DefaultAzureCredential
from azure.mgmt.compute import ComputeManagementClient
from azure.mgmt.subscription import SubscriptionClient
from azure.core.exceptions import HttpResponseError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('vm_sku_capacity_monitor')

try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Azure VM SKU Capacity Monitor')
    parser.add_argument('--region', type=str, default='eastus2',
                        help='Azure region to check (default: eastus2)')
    parser.add_argument('--sku', type=str, default='Standard_D16ds_v5',
                        help='VM SKU to check (default: Standard_D16ds_v5)')
    parser.add_argument('--log-analytics', action='store_true',
                        help='Enable logging to Azure Log Analytics')
    parser.add_argument('--endpoint', type=str,
                        help='Azure Monitor Data Collection Endpoint URI')
    parser.add_argument('--rule-id', type=str,
                        help='Azure Monitor Data Collection Rule ID')
    parser.add_argument('--stream-name', type=str, default='Custom-VMSKUCapacity_CL',
                        help='Azure Monitor Log Analytics stream name')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    parser.add_argument('--config', type=str,
                        help='Path to configuration file')
    parser.add_argument('--subscription-id', type=str,
                        help='Azure Subscription ID')
    
    return parser.parse_args()

def load_configuration(args):
    """Load configuration from file or command line arguments."""
    config = {
        'region': args.region,
        'target_sku': args.sku,
        'check_zones': True,
        'subscription_id': args.subscription_id,
        'log_analytics': {
            'enabled': args.log_analytics,
            'endpoint': args.endpoint,
            'rule_id': args.rule_id,
            'stream_name': args.stream_name
        },
        'check_interval_minutes': 15
    }
    
    if args.config:
        try:
            with open(args.config, 'r') as f:
                file_config = json.load(f)
                logger.info(f"Configuration loaded from {args.config}")
                
                # Update config with file values
                config['region'] = file_config.get('region', config['region'])
                config['target_sku'] = file_config.get('target_sku', config['target_sku'])
                config['check_zones'] = file_config.get('check_zones', config['check_zones'])
                config['check_interval_minutes'] = file_config.get('check_interval_minutes', config['check_interval_minutes'])
                config['subscription_id'] = file_config.get('subscription_id', config['subscription_id'])
                
                # Update Log Analytics config
                if 'log_analytics' in file_config:
                    config['log_analytics']['enabled'] = file_config['log_analytics'].get('enabled', config['log_analytics']['enabled'])
                    config['log_analytics']['endpoint'] = file_config['log_analytics'].get('endpoint', config['log_analytics']['endpoint'])
                    config['log_analytics']['rule_id'] = file_config['log_analytics'].get('rule_id', config['log_analytics']['rule_id'])
                    config['log_analytics']['stream_name'] = file_config['log_analytics'].get('stream_name', config['log_analytics']['stream_name'])
        except Exception as e:
            logger.error(f"Error loading configuration from {args.config}: {str(e)}")
            logger.info("Using default configuration")
    
    # Command line arguments override config file
    if args.region:
        config['region'] = args.region
    if args.sku:
        config['target_sku'] = args.sku
    if args.log_analytics:
        config['log_analytics']['enabled'] = True
    if args.endpoint:
        config['log_analytics']['endpoint'] = args.endpoint
    if args.rule_id:
        config['log_analytics']['rule_id'] = args.rule_id
    if args.stream_name:
        config['log_analytics']['stream_name'] = args.stream_name
    if args.subscription_id:
        config['subscription_id'] = args.subscription_id
    
    # Auto-detect subscription ID if not provided
    if not config.get('subscription_id'):
        config['subscription_id'] = get_subscription_id(config)
        
    return config

def get_subscription_id(config):
    """Automatically detect the subscription ID using multiple methods."""
    subscription_id = None
    
    # Method 1: Try to extract from rule_id in config
    if config.get('log_analytics', {}).get('rule_id'):
        rule_id = config['log_analytics']['rule_id']
        match = re.search(r'/subscriptions/([^/]+)/', rule_id)
        if match:
            subscription_id = match.group(1)
            logger.info(f"Extracted subscription ID from rule_id: {subscription_id}")
            return subscription_id
    
    # Method 2: Try to get from Azure CLI
    try:
        result = subprocess.run(
            "az account show --query id -o tsv",
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        subscription_id = result.stdout.strip()
        if subscription_id:
            logger.info(f"Retrieved subscription ID from Azure CLI: {subscription_id}")
            return subscription_id
    except Exception as e:
        logger.debug(f"Could not get subscription ID from Azure CLI: {str(e)}")
    
    # Method 3: Try to get from DefaultAzureCredential
    try:
        credential = DefaultAzureCredential()
        subscription_client = SubscriptionClient(credential)
        subscriptions = list(subscription_client.subscriptions.list())
        if subscriptions:
            subscription_id = subscriptions[0].subscription_id
            logger.info(f"Retrieved subscription ID from Azure SDK: {subscription_id}")
            return subscription_id
    except Exception as e:
        logger.debug(f"Could not get subscription ID from Azure SDK: {str(e)}")
    
    if not subscription_id:
        logger.warning("Could not automatically detect subscription ID. Please provide it manually.")
    
    return subscription_id

def check_sku_availability(compute_client, region, target_sku, check_zones=True):
    """Check if a specific VM SKU is available in the given region."""
    # Get all SKUs
    skus = list(compute_client.resource_skus.list())
    
    # Find the target SKU in the specified region
    target_sku_info = None
    for sku in skus:
        if sku.name.lower() == target_sku.lower() and any(loc.lower() == region.lower() for loc in sku.locations):
            target_sku_info = sku
            break
    
    if not target_sku_info:
        logger.warning(f"SKU {target_sku} not found in region {region}")
        return False, "NotFound", [], {}, []
    
    # Check availability
    is_available = True
    restriction_reason = None
    restrictions = []
    
    for restriction in target_sku_info.restrictions:
        if any(value.lower() == region.lower() for value in restriction.restriction_info.locations):
            is_available = False
            restriction_reason = restriction.reason_code
            restrictions.append({
                'type': restriction.type,
                'reason': restriction.reason_code,
                'values': restriction.restriction_info.locations
            })
    
    # Get zone availability
    zones = []
    if check_zones and hasattr(target_sku_info, 'location_info'):
        for location_info in target_sku_info.location_info:
            if location_info.location.lower() == region.lower() and hasattr(location_info, 'zones'):
                zones = location_info.zones
    
    # Get SKU specifications
    specifications = {}
    if hasattr(target_sku_info, 'capabilities'):
        for capability in target_sku_info.capabilities:
            specifications[capability.name] = capability.value
    
    # Find alternative SKUs
    alternative_skus = []
    if not is_available:
        for sku in skus:
            # Skip if not a VM SKU or same as target
            if sku.resource_type != 'virtualMachines' or sku.name == target_sku:
                continue
            
            # Check if available in the region
            if not any(loc.lower() == region.lower() for loc in sku.locations):
                continue
            
            # Check if restricted in the region
            is_restricted = False
            for restriction in sku.restrictions:
                if any(value.lower() == region.lower() for value in restriction.restriction_info.locations):
                    is_restricted = True
                    break
            
            if is_restricted:
                continue
            
            # Get specifications
            alt_specs = {}
            if hasattr(sku, 'capabilities'):
                for capability in sku.capabilities:
                    alt_specs[capability.name] = capability.value
            
            # Calculate similarity score
            similarity = calculate_similarity(specifications, alt_specs)
            
            if similarity >= 80:  # Only include if at least 80% similar
                alternative_skus.append({
                    'name': sku.name,
                    'vcpus': alt_specs.get('vCPUs', 'Unknown'),
                    'memory': alt_specs.get('MemoryGB', 'Unknown'),
                    'family': sku.family,
                    'similarity': similarity
                })
        
        # Sort by similarity (highest first)
        alternative_skus.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Limit to top 5
        alternative_skus = alternative_skus[:5]
    
    logger.info(f"Availability check result: {is_available}, Reason: {restriction_reason}")
    
    return is_available, restriction_reason, zones, specifications, alternative_skus

def calculate_similarity(specs1, specs2):
    """Calculate similarity percentage between two SKU specifications."""
    # Key specifications to compare
    key_specs = ['vCPUs', 'MemoryGB', 'MaxDataDiskCount', 'PremiumIO', 'AcceleratedNetworkingEnabled']
    
    # Count matches
    matches = 0
    total = 0
    
    for key in key_specs:
        if key in specs1 and key in specs2:
            total += 1
            if specs1[key] == specs2[key]:
                matches += 1
    
    # Calculate percentage
    if total == 0:
        return 0
    
    return int((matches / total) * 100)

def display_results_rich(region, target_sku, is_available, restriction_reason, zones, specifications, alternative_skus, subscription_name, subscription_id):
    """Display results using rich formatting."""
    console = Console()
    
    # Create header
    console.print(f"[bold white on blue]{'AZURE VM SKU CAPACITY MONITOR - ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'):^150}[/]")
    
    # Create summary table
    console.print()
    console.print(f"  [bold]Status[/]         {'AVAILABLE' if is_available else 'NOT AVAILABLE'}")
    console.print(f"  [bold]SKU[/]            {target_sku}")
    console.print(f"  [bold]Region[/]         {region}")
    console.print(f"  [bold]Subscription[/]   {subscription_name} ({subscription_id})")
    if not is_available:
        console.print(f"  [bold]Details[/]        SKU {target_sku} is not available in region {region}")
    
    # Display zones
    console.print()
    console.print("[bold]Available[/]")
    console.print("  [bold]Zones[/]")
    
    if zones:
        zone_table = Table(show_header=True, header_style="bold", box=box.SIMPLE)
        zone_table.add_column("Zone")
        for zone in zones:
            zone_table.add_row(zone)
        console.print(zone_table)
    else:
        console.print("  None")
    
    # Display restrictions
    if not is_available:
        console.print()
        console.print("[bold]Restrictions[/]".center(50))
        
        restrictions_table = Table(show_header=True, header_style="bold", box=box.SIMPLE)
        restrictions_table.add_column("Type", style="dim")
        restrictions_table.add_column("Reason", style="dim")
        restrictions_table.add_column("Affected Values", style="dim")
        
        restrictions_table.add_row("Zone", restriction_reason, region)
        console.print(restrictions_table)
    
    # Display specifications
    console.print()
    console.print("[bold]VM SKU Specifications[/]".center(50))
    
    specs_table = Table(show_header=True, header_style="bold", box=box.SIMPLE)
    specs_table.add_column("Property", style="dim")
    specs_table.add_column("Value", style="dim")
    
    for key, value in specifications.items():
        specs_table.add_row(key, str(value))
    
    console.print(specs_table)
    
    # Display alternative SKUs
    if alternative_skus:
        console.print()
        console.print("[bold]Alternative SKUs[/]".center(50))
        
        alt_table = Table(show_header=True, header_style="bold", box=box.SIMPLE)
        alt_table.add_column("SKU Name", style="dim")
        alt_table.add_column("vCPUs", style="dim")
        alt_table.add_column("Memory (GB)", style="dim")
        alt_table.add_column("Family", style="dim")
        alt_table.add_column("Similarity", style="dim")
        
        for sku in alternative_skus:
            alt_table.add_row(
                sku['name'],
                str(sku['vcpus']),
                str(sku['memory']),
                sku['family'],
                f"{sku['similarity']}%"
            )
        
        console.print(alt_table)

def display_results_text(region, target_sku, is_available, restriction_reason, zones, specifications, alternative_skus, subscription_name, subscription_id):
    """Display results using plain text formatting."""
    print("\n" + "=" * 80)
    print(f"AZURE VM SKU CAPACITY MONITOR - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    print(f"\nStatus:       {'AVAILABLE' if is_available else 'NOT AVAILABLE'}")
    print(f"SKU:          {target_sku}")
    print(f"Region:       {region}")
    print(f"Subscription: {subscription_name} ({subscription_id})")
    if not is_available:
        print(f"Details:      SKU {target_sku} is not available in region {region}")
    
    print("\nAvailable Zones:")
    if zones:
        for zone in zones:
            print(f"  - {zone}")
    else:
        print("  None")
    
    if not is_available:
        print("\nRestrictions:")
        print(f"  Type:           Zone")
        print(f"  Reason:         {restriction_reason}")
        print(f"  Affected Values: {region}")
    
    print("\nVM SKU Specifications:")
    for key, value in specifications.items():
        print(f"  {key}: {value}")
    
    if alternative_skus:
        print("\nAlternative SKUs:")
        for sku in alternative_skus:
            print(f"  - {sku['name']} (vCPUs: {sku['vcpus']}, Memory: {sku['memory']} GB, Family: {sku['family']}, Similarity: {sku['similarity']}%)")

def log_to_azure_monitor(data, log_analytics_config):
    """Log data to Azure Monitor."""
    try:
        # Import Azure Monitor Ingestion client
        from azure.monitor.ingestion import LogsIngestionClient
        
        # Initialize the logs ingestion client
        credential = DefaultAzureCredential()
        logs_client = LogsIngestionClient(endpoint=log_analytics_config['endpoint'], credential=credential)
        
        # Prepare the log entry
        log_entry = {
            "TimeGenerated": datetime.datetime.utcnow().isoformat(),
            "sku_name": data['sku'],
            "region": data['region'],
            "subscription_name": data['subscription_name'],
            "subscription_id": data['subscription_id'],
            "is_available": data['is_available'],
            "restriction_reason": data['restriction_reason'] or "",
            "zones": ",".join(data['zones']),
            "vcpus": data['specifications'].get('vCPUs', ""),
            "memory_gb": data['specifications'].get('MemoryGB', ""),
            "alternative_skus": ",".join([sku['name'] for sku in data['alternative_skus']])
        }
        
        # Upload the log entry
        logs_client.upload(
            rule_id=log_analytics_config['rule_id'],
            stream_name=log_analytics_config['stream_name'],
            logs=[log_entry]
        )
        
        logger.info("Successfully logged to Azure Monitor")
        return True
    except ImportError:
        logger.error("Azure Monitor Ingestion client not installed. Install with: pip install azure-monitor-ingestion")
        return False
    except HttpResponseError as e:
        logger.error(f"Error logging to Azure Monitor: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Error logging to Azure Monitor: {str(e)}")
        return False

def main():
    """Main function."""
    # Parse arguments
    args = parse_arguments()
    
    # Set debug logging if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logging.getLogger('azure').setLevel(logging.DEBUG)
    
    # Load configuration
    config = load_configuration(args)
    
    # Log start
    logger.info(f"Starting VM SKU capacity monitoring for {config['target_sku']} in {config['region']}")
    
    try:
        # Initialize Azure clients
        credential = DefaultAzureCredential()
        compute_client = ComputeManagementClient(credential, subscription_id=config['subscription_id'])
        subscription_client = SubscriptionClient(credential)
        
        # Get subscription details
        subscriptions = list(subscription_client.subscriptions.list())
        subscription_name = subscriptions[0].display_name if subscriptions else "Unknown"
        subscription_id = subscriptions[0].subscription_id if subscriptions else config['subscription_id']
        
        # Check SKU availability
        is_available, restriction_reason, zones, specifications, alternative_skus = check_sku_availability(
            compute_client,
            config['region'],
            config['target_sku'],
            config['check_zones']
        )
        
        # Display results
        if not is_available:
            logger.warning(f"SKU {config['target_sku']} is not available in region {config['region']}")
        
        # Prepare result data
        result_data = {
            'sku': config['target_sku'],
            'region': config['region'],
            'subscription_name': subscription_name,
            'subscription_id': subscription_id,
            'is_available': is_available,
            'restriction_reason': restriction_reason,
            'zones': zones,
            'specifications': specifications,
            'alternative_skus': alternative_skus
        }
        
        # Display results
        if RICH_AVAILABLE:
            display_results_rich(
                config['region'],
                config['target_sku'],
                is_available,
                restriction_reason,
                zones,
                specifications,
                alternative_skus,
                subscription_name,
                subscription_id
            )
        else:
            display_results_text(
                config['region'],
                config['target_sku'],
                is_available,
                restriction_reason,
                zones,
                specifications,
                alternative_skus,
                subscription_name,
                subscription_id
            )
        
        # Log to Azure Monitor if enabled
        if config['log_analytics']['enabled']:
            if not config['log_analytics']['endpoint'] or not config['log_analytics']['rule_id']:
                logger.error("Log Analytics endpoint and rule ID are required for Azure Monitor logging")
            else:
                try:
                    log_to_azure_monitor(result_data, config['log_analytics'])
                except Exception as e:
                    logger.error(f"Failed to log to Azure Monitor: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error monitoring VM SKU capacity: {str(e)}")
        if args.debug:
            import traceback
            traceback.print_exc()
    
    logger.info("VM SKU capacity monitoring completed")

if __name__ == "__main__":
    main()