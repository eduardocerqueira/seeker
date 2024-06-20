#date: 2024-06-20T16:46:08Z
#url: https://api.github.com/gists/28a8dedae447c6ca5d8a2ff29ee3dd73
#owner: https://api.github.com/users/AndrewAltimit

import json
import sys
import boto3
from botocore.exceptions import ClientError
import subprocess
import os
import yaml

def run_command(command, working_directory):
    # Ensure the working directory exists
    if not os.path.exists(working_directory):
        raise ValueError(f"Working directory does not exist: {working_directory}")

    # Create a new process
    process = subprocess.Popen(
        command.split(),
        cwd=working_directory,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Capture output in real-time
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())

    # Wait for the process to complete and get the return code
    return_code = process.poll()

    # Capture any remaining output
    stdout, stderr = process.communicate()

    if return_code != 0:
        print(f"Error occurred. Return code: {return_code}")
        print(f"Error output: {stderr}")

    return return_code

def load_config(config_path='config.yaml'):
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' not found.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML configuration: {e}")
        sys.exit(1)

def download_from_s3(bucket, key, local_path):
    s3 = boto3.client('s3')
    try:
        s3.download_file(bucket, key, local_path)
        print(f"Downloaded {bucket}/{key} to {local_path}")
    except ClientError as e:
        print(f"Error downloading from S3: {e}")
        sys.exit(1)

def upload_to_s3(local_path, bucket, key):
    s3 = boto3.client('s3')
    try:
        s3.upload_file(local_path, bucket, key)
        print(f"Uploaded {local_path} to {bucket}/{key}")
    except ClientError as e:
        print(f"Error uploading to S3: {e}")
        sys.exit(1)

def delete_from_s3(bucket, key):
    s3 = boto3.client('s3')
    try:
        s3.delete_object(Bucket=bucket, Key=key)
        print(f"Deleted {bucket}/{key} from S3")
    except ClientError as e:
        print(f"Error deleting from S3: {e}")
        sys.exit(1)

def delete_from_dynamodb(table, key, pattern):
    dynamodb = boto3.client('dynamodb')
    try:
        # Scan the table for items matching the pattern
        response = dynamodb.scan(
            TableName=table,
            FilterExpression=f"contains(#lockid, :pattern)",
            ExpressionAttributeNames={"#lockid": key},
            ExpressionAttributeValues={":pattern": {"S": pattern}}
        )

        # Delete matching items
        for item in response.get('Items', []):
            dynamodb.delete_item(
                TableName=table,
                Key={key: item[key]}
            )
            print(f"Deleted item with {key}: {item[key]['S']} from DynamoDB table {table}")

    except ClientError as e:
        print(f"Error deleting from DynamoDB: {e}")
        sys.exit(1)

def merge_tfstates(state1_path, state2_path, output_path):
    with open(state1_path, 'r') as f:
        state1 = json.load(f)
    with open(state2_path, 'r') as f:
        state2 = json.load(f)

    # Check for resource naming conflicts
    resources1 = {(r['type'], r['name']): r for r in state1['resources']}
    resources2 = {(r['type'], r['name']): r for r in state2['resources']}

    conflicts = set(resources1.keys()) & set(resources2.keys())
    if conflicts:
        raise ValueError(f"Resource naming conflicts found: {conflicts}")

    merged_resources = state1['resources'] + state2['resources']
    merged_outputs = {**state1.get('outputs', {}), **state2.get('outputs', {})}

    merged_state = {
        "version": max(state1.get('version', 0), state2.get('version', 0)),
        "terraform_version": state1.get('terraform_version', state2.get('terraform_version', '')),
        "serial": max(state1.get('serial', 0), state2.get('serial', 0)) + 1,
        "lineage": state1.get('lineage', ''),
        "outputs": merged_outputs,
        "resources": merged_resources
    }

    with open(output_path, 'w') as f:
        json.dump(merged_state, f, indent=2)

    print(f"Merged state file created at: {output_path}")

if __name__ == "__main__":
    # Load Configuration
    config = load_config()
    tfstate_bucket = config['tfstate_bucket']
    tfstate_root = config['tfstate_root']
    local_modules_path = config['local_modules_path']
    destination_module = config['destination_module']
    state_filename = config['state_filename']
    dynamodb_lock_table = config['dynamodb_lock_table']
    dynamodb_lock_pattern = config['dynamodb_lock_pattern']
    dry_run = config['dry_run']

    # Create a temporary directory for backing up tfstates
    backup_dir = "backup_tfstates"
    os.makedirs(backup_dir, exist_ok=True)

    # Download state files from S3
    local_state1_path = os.path.join(backup_dir, "state1.tfstate")
    local_state2_path = os.path.join(backup_dir, "state2.tfstate")
    download_from_s3(bucket=tfstate_bucket, key=f"{tfstate_root}/{config['source_module_1']}/{state_filename}", local_path=local_state1_path)
    download_from_s3(bucket=tfstate_bucket, key=f"{tfstate_root}/{config['source_module_2']}/{state_filename}", local_path=local_state2_path)
    print(f"Local backups of state files are available in the '{backup_dir}' directory.")

    try:
        # Generate merged state
        destination_path = os.path.join(local_modules_path, destination_module)
        local_merged_state = os.path.join(destination_path, "merged_state.tfstate")
        merge_tfstates(local_state1_path, local_state2_path, local_merged_state)

        # Remove old state file on s3 for the destination_module
        if dry_run:
            print(f"[Dry Run] Delete S3 file s3://{tfstate_bucket}/{tfstate_root}/{destination_module}/{state_filename}")
        else:
            delete_from_s3(bucket=tfstate_bucket, key=f"{tfstate_root}/{destination_module}/{state_filename}")

        # Remove dynamodb entries in table <dynamodb_lock_table> where LockID contains <dynamodb_lock_pattern>
        if dry_run:
            print(f"[Dry Run] Delete DynamoDB entries in {dynamodb_lock_table} containing LockID {dynamodb_lock_pattern}")
        else:
            delete_from_dynamodb(table=dynamodb_lock_table, key="LockID", pattern=dynamodb_lock_pattern)

        # Initialize tfstate for this module
        if dry_run:
            print(f"[Dry Run] terragrunt init --reconfigure\n\t(Working Directory: {destination_path})")
        else:
            run_command("terragrunt init --reconfigure", destination_path)

        # Push merged terraform state file
        if dry_run:
            print(f"[Dry Run] terragrunt state push {local_merged_state}\n\t(Working Directory: {destination_path})")
        else:
            run_command(f"terragrunt state push {local_merged_state}", destination_path)

    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print("Terraform state merge completed successfully.")