#date: 2025-05-22T17:07:22Z
#url: https://api.github.com/gists/76b0e2e96f288a9b2635233570f5d4d7
#owner: https://api.github.com/users/ricmmartins

#!/usr/bin/env python
"""
Azure VM SKU Capacity Monitor - Log Analytics Setup

This script automates the creation of:
  • Resource Group
  • Log Analytics Workspace (and waits for it to become active)
  • Data Collection Endpoint
  • Data Collection Rule
  • Custom table in the workspace

It then emits a `config.json` for `monitor_vm_sku_capacity_terminal.py`.
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("log_analytics_setup")


def run_command(cmd: str) -> str:
    """Run a shell command, returning stdout or raising on failure."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {cmd}")
        logger.error(e.stderr.strip())
        raise


def parse_arguments():
    p = argparse.ArgumentParser(
        description="Setup Log Analytics for VM SKU Capacity Monitor"
    )
    p.add_argument(
        "--resource-group",
        "-g",
        default="vm-sku-monitor-rg",
        help="Resource group name",
    )
    p.add_argument(
        "--location", "-l", default="eastus2", help="Azure region (e.g. eastus2)"
    )
    p.add_argument(
        "--workspace",
        "-w",
        default="vmskumonitor-workspace",
        help="Log Analytics workspace name",
    )
    p.add_argument(
        "--dce",
        default="vmskumonitor-dce",
        help="Data Collection Endpoint name",
    )
    p.add_argument(
        "--dcr",
        default="vmskumonitor-dcr",
        help="Data Collection Rule name",
    )
    p.add_argument(
        "--table",
        "-t",
        default="VMSKUCapacity",
        help="Base name for custom table (suffix _CL added)",
    )
    p.add_argument(
        "--config",
        "-c",
        default="config.json",
        help="Output configuration file path",
    )
    return p.parse_args()


def ensure_rg(rg: str, loc: str):
    logger.info(f"Ensuring resource group {rg} exists in {loc}")
    try:
        run_command(f"az group show -n {rg}")
        logger.info(f"Resource group {rg} already exists.")
    except:
        run_command(f"az group create -n {rg} -l {loc}")
        logger.info(f"Resource group {rg} created.")


def ensure_workspace(rg: str, ws: str, loc: str):
    logger.info(f"Ensuring Log Analytics workspace {ws}")
    try:
        run_command(
            f"az monitor log-analytics workspace show "
            f"-g {rg} -n {ws}"
        )
        logger.info(f"Workspace {ws} already exists.")
    except:
        run_command(
            f"az monitor log-analytics workspace create "
            f"-g {rg} -n {ws} -l {loc}"
        )
        logger.info(f"Workspace {ws} created.")
    wait_for_workspace(rg, ws)


def wait_for_workspace(rg: str, ws: str, timeout: int = 300, interval: int = 10):
    logger.info(f"Waiting up to {timeout}s for workspace {ws} to become active…")
    elapsed = 0
    while elapsed < timeout:
        state = run_command(
            f"az monitor log-analytics workspace show "
            f"-g {rg} -n {ws} --query provisioningState -o tsv"
        ).strip().lower()
        if state == "succeeded":
            logger.info("Workspace is active.")
            return
        logger.info(f"Current state: {state!r}; retrying in {interval}s…")
        time.sleep(interval)
        elapsed += interval
    logger.warning(f"Workspace did not become active within {timeout}s; continuing.")


def ensure_dce(rg: str, dce: str, loc: str) -> str:
    logger.info(f"Ensuring Data Collection Endpoint {dce}")
    try:
        run_command(f"az monitor data-collection endpoint show -g {rg} -n {dce}")
        logger.info(f"DCE {dce} already exists.")
    except:
        run_command(
            f"az monitor data-collection endpoint create "
            f"-g {rg} -n {dce} -l {loc} --public-network-access Enabled"
        )
        logger.info(f"DCE {dce} created.")
    out = run_command(f"az monitor data-collection endpoint show -g {rg} -n {dce} -o json")
    return json.loads(out)["logsIngestion"]["endpoint"]


def deploy_custom_table(rg: str, ws: str, table: str):
    # Wait once more in case ingestion APIs lag behind provisioningState
    logger.info("Re-checking workspace readiness before custom table deployment…")
    wait_for_workspace(rg, ws, timeout=180, interval=15)

    arm = {
        "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",
        "contentVersion": "1.0.0.0",
        "parameters": {
            "workspaceName": {"type": "string"},
            "tableName": {"type": "string"},
        },
        "resources": [
            {
                "type": "Microsoft.OperationalInsights/workspaces/tables",
                "apiVersion": "2021-12-01-preview",
                "name": "[concat(parameters('workspaceName'), '/', parameters('tableName'), '_CL')]",
                "properties": {
                    "schema": {
                        "name": "[concat(parameters('tableName'), '_CL')]",
                        "columns": [
                            {"name": "TimeGenerated", "type": "datetime"},
                            {"name": "sku_name", "type": "string"},
                            {"name": "region", "type": "string"},
                            {"name": "subscription_name", "type": "string"},
                            {"name": "subscription_id", "type": "string"},
                            {"name": "is_available", "type": "boolean"},
                            {"name": "restriction_reason", "type": "string"},
                            {"name": "zones", "type": "string"},
                            {"name": "vcpus", "type": "string"},
                            {"name": "memory_gb", "type": "string"},
                            {"name": "alternative_skus", "type": "string"},
                        ],
                    }
                },
            }
        ],
    }

    fn = f"custom-table-{int(time.time())}.json"
    with open(fn, "w") as f:
        json.dump(arm, f, indent=2)
    logger.info(f"Deploying custom table {table}_CL via ARM template")
    run_command(
        f"az deployment group create -g {rg} "
        f"--template-file {fn} "
        f"--parameters workspaceName={ws} tableName={table}"
    )
    os.remove(fn)
    logger.info("Custom table created.")


def deploy_dcr(rg: str, dcr: str, loc: str, dce_uri: str, ws: str, table: str) -> str:
    """
    Create or verify a Data Collection Rule that sends Custom-<table>_CL
    to the workspace. Returns the ImmutableId.
    """
    logger.info(f"Ensuring Data Collection Rule {dcr}")
    try:
        run_command(f"az monitor data-collection rule show -g {rg} -n {dcr}")
        logger.info(f"DCR {dcr} already exists.")
    except:
        # gather resource IDs
        ws_id = run_command(f"az monitor log-analytics workspace show -g {rg} -n {ws} -o json")
        ws_id = json.loads(ws_id)["id"]
        dce_id = run_command(f"az monitor data-collection endpoint show -g {rg} -n {dcr.replace('-dcr','-dce')} -o json")
        dce_id = json.loads(dce_id)["id"]

        # build ARM
        arm = {
            "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",
            "contentVersion": "1.0.0.0",
            "parameters": {
                "dcrName": {"type": "string"},
                "location": {"type": "string"},
                "dceId": {"type": "string"},
                "workspaceId": {"type": "string"},
                "streamName": {"type": "string"},
            },
            "resources": [
                {
                    "type": "Microsoft.Insights/dataCollectionRules",
                    "apiVersion": "2021-09-01-preview",
                    "name": "[parameters('dcrName')]",
                    "location": "[parameters('location')]",
                    "properties": {
                        "dataCollectionEndpointId": "[parameters('dceId')]",
                        "streamDeclarations": {
                            "[parameters('streamName')]": {
                                "columns": [
                                    {"name": "TimeGenerated", "type": "datetime"},
                                    {"name": "sku_name", "type": "string"},
                                    {"name": "region", "type": "string"},
                                    {"name": "subscription_name", "type": "string"},
                                    {"name": "subscription_id", "type": "string"},
                                    {"name": "is_available", "type": "boolean"},
                                    {"name": "restriction_reason", "type": "string"},
                                    {"name": "zones", "type": "string"},
                                    {"name": "vcpus", "type": "string"},
                                    {"name": "memory_gb", "type": "string"},
                                    {"name": "alternative_skus", "type": "string"},
                                ]
                            }
                        },
                        "destinations": {
                            "logAnalytics": [
                                {
                                    "workspaceResourceId": "[parameters('workspaceId')]",
                                    "name": "la-destination",
                                }
                            ]
                        },
                        "dataFlows": [
                            {
                                "streams": ["[parameters('streamName')]"],
                                "destinations": ["la-destination"],
                            }
                        ],
                    },
                }
            ],
        }
        fn = f"dcr-{int(time.time())}.json"
        with open(fn, "w") as f:
            json.dump(arm, f, indent=2)
        run_command(
            f"az deployment group create -g {rg} "
            f"--template-file {fn} "
            f"--parameters "
            f"dcrName={dcr} location={loc} "
            f"dceId={dce_id} workspaceId={ws_id} "
            f"streamName=Custom-{table}_CL"
        )
        os.remove(fn)
        logger.info(f"DCR {dcr} created.")

    # return immutableId
    out = run_command(f"az monitor data-collection rule show -g {rg} -n {dcr} -o json")
    return json.loads(out)["immutableId"]


def write_config(path: str, dce_uri: str, dcr_id: str, table: str, loc: str):
    cfg = {
        "region": loc,
        "target_sku": "Standard_D16ds_v5",
        "check_zones": True,
        "log_analytics": {
            "enabled": True,
            "endpoint": dce_uri,
            "rule_id": dcr_id,
            "stream_name": f"Custom-{table}_CL",
        },
        "check_interval_minutes": 15,
    }
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    logger.info(f"Wrote configuration to {path}")


def main():
    args = parse_arguments()
    logger.info("Starting Log Analytics setup…")

    ensure_rg(args.resource_group, args.location)
    ensure_workspace(args.resource_group, args.workspace, args.location)
    
    # Create the Data Collection Endpoint
    dce_uri = ensure_dce(args.resource_group, args.dce, args.location)
    
    # IMPORTANT: Create the custom table BEFORE the DCR
    # This fixes the "InvalidOutputTable" error
    deploy_custom_table(args.resource_group, args.workspace, args.table)
    
    # Now create the Data Collection Rule that references the custom table
    dcr_id = deploy_dcr(
        args.resource_group, args.dcr, args.location, dce_uri, args.workspace, args.table
    )

    write_config(args.config, dce_uri, dcr_id, args.table, args.location)

    logger.info("Log Analytics setup completed successfully!")


if __name__ == "__main__":
    main()