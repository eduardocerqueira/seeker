#date: 2024-09-20T16:46:18Z
#url: https://api.github.com/gists/a12cf70804125557a0821cca20be746e
#owner: https://api.github.com/users/Rajchowdhury420

import boto3
from rich.table import Table
from rich.console import Console

# Initialize boto3 clients
cloudformation = boto3.client('cloudformation')
console = Console()

def list_stacksets():
    """
    List all StackSets in the account.
    """
    paginator = cloudformation.get_paginator('list_stack_sets')
    stacksets = []

    for page in paginator.paginate():
        stacksets.extend(page['Summaries'])

    return stacksets

def detect_stackset_drift(stackset_name):
    """
    Detect drift for a specific StackSet.
    """
    try:
        response = cloudformation.detect_stack_set_drift(StackSetName=stackset_name)
        return response['StackSetDriftDetectionId']
    except Exception as e:
        console.print(f"Error detecting drift for {stackset_name}: {e}")
        return None

def describe_stackset_drift_detection(drift_detection_id):
    """
    Describe the drift detection results for a StackSet.
    """
    try:
        response = cloudformation.describe_stack_set_drift_detection_status(
            StackSetDriftDetectionId=drift_detection_id
        )
        return response
    except Exception as e:
        console.print(f"Error describing drift detection: {e}")
        return None

def detect_drift_in_all_stacksets():
    """
    Detect drift for all StackSets and display in a formatted table.
    """
    stacksets = list_stacksets()

    if not stacksets:
        console.print("No StackSets found in the account.")
        return

    table = Table(title="StackSet Drift Status")

    # Add columns to the table
    table.add_column("StackSet Name", justify="left", style="cyan", no_wrap=True)
    table.add_column("Drift Detection ID", justify="left", style="yellow")
    table.add_column("Drift Status", justify="left", style="green")
    table.add_column("Detection Status", justify="left", style="magenta")
    table.add_column("Drifted Stacks", justify="right", style="red")

    # Iterate over stacksets to detect drift
    for stackset in stacksets:
        stackset_name = stackset['StackSetName']
        drift_detection_id = detect_stackset_drift(stackset_name)

        if drift_detection_id:
            drift_status = describe_stackset_drift_detection(drift_detection_id)

            # Add stackset details to the table
            table.add_row(
                stackset_name,
                drift_detection_id,
                drift_status['StackSetDriftStatus'],
                drift_status['DetectionStatus'],
                str(drift_status['DriftedStackInstancesCount'])
            )
        else:
            table.add_row(stackset_name, "N/A", "N/A", "N/A", "N/A")

    # Print the table using rich
    console.print(table)

if __name__ == "__main__":
    detect_drift_in_all_stacksets()