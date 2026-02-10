#date: 2026-02-10T17:51:27Z
#url: https://api.github.com/gists/1d66508f30e683747f19537763d1f4c9
#owner: https://api.github.com/users/kwadie

# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import csv
import argparse
import sys
import logging
import time
from datetime import datetime, timedelta, timezone
from google.cloud import bigquery_datatransfer
from google.protobuf import timestamp_pb2

# Configuration
DATA_SOURCE_ID = "cross_region_copy"

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def create_transfer_config(client, source_project_id, source_dataset_id, target_project_id, target_dataset_id, service_account, timestamp_prefix, dry_run=False):
    """Creates a BigQuery Data Transfer Service configuration."""
    
    parent = f"projects/{target_project_id}"
    
    transfer_config = bigquery_datatransfer.TransferConfig(
        destination_dataset_id=target_dataset_id,
        display_name=f"{timestamp_prefix} - Copy {source_dataset_id} from {source_project_id}",
        data_source_id=DATA_SOURCE_ID,
        params={
            "source_dataset_id": source_dataset_id,
            "source_project_id": source_project_id,
            "overwrite_destination_table": "true"
        },
        email_preferences=bigquery_datatransfer.EmailPreferences(enable_failure_email=True),
        schedule_options=bigquery_datatransfer.ScheduleOptions(disable_auto_scheduling=True)
    )

    if dry_run:
        dummy_name = f"projects/{target_project_id}/locations/us/transferConfigs/dry-run-{target_dataset_id}"
        logging.info(f"[DRY RUN] Would create transfer config for target dataset '{target_dataset_id}' in project '{target_project_id}'.")
        logging.info(f"          - Display Name: {transfer_config.display_name}")
        logging.info(f"          - Source Project: {source_project_id}")
        logging.info(f"          - Source Dataset: {source_dataset_id}")
        logging.info(f"          - Service Account: {service_account}")
        logging.info(f"          - Email Preferences: Failure emails enabled")
        logging.info(f"          - Overwrite Destination: True")
        return dummy_name

    try:
        request = bigquery_datatransfer.CreateTransferConfigRequest(
            parent=parent,
            transfer_config=transfer_config,
            service_account_name=service_account
        )
        response = client.create_transfer_config(request=request)
        logging.info(f"Created config '{response.name}' for target dataset '{target_dataset_id}'")
        return response.name
    except Exception as e:
        logging.error(f"Failed to create config for target dataset '{target_dataset_id}': {e}")
        return None

def parse_csv(file_path):
    jobs = []
    try:
        with open(file_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            
            # Normalize field names to handle potential whitespace or case issues
            if reader.fieldnames:
                 reader.fieldnames = [name.strip().lower() for name in reader.fieldnames]

            for row in reader:
                # Map expected fields
                # Expected: source_project_id, source_dataset_id, target_project_id, target_dataset_id, target_project_service_account_email
                try:
                    jobs.append({
                        "source_project_id": row.get('source_project_id', '').strip(),
                        "source_dataset_id": row.get('source_dataset_id', '').strip(),
                        "target_project_id": row.get('target_project_id', '').strip(),
                        "target_dataset_id": row.get('target_dataset_id', '').strip(),
                        "service_account": row.get('target_project_service_account_email', '').strip()
                    })
                except AttributeError:
                    # Handle cases where row might be None or malformed
                    continue
                
                # Check for empty required values
                if not all(jobs[-1].values()):
                    logging.warning(f"Skipping incomplete row: {row}")
                    jobs.pop()
                    continue

    except Exception as e:
        logging.error(f"Error reading CSV file: {e}")
        sys.exit(1)
        
    return jobs

def main():
    parser = argparse.ArgumentParser(description='Automate BigQuery Dataset Copy DTS creation.')
    parser.add_argument('--input', required=True, help='Path to input CSV file')
    parser.add_argument('--dry-run', action='store_true', help='Simulate job creation without executing API calls')
    args = parser.parse_args()

    setup_logging()
    
    jobs = parse_csv(args.input)
    if not jobs:
        logging.info("No jobs found in input file.")
        return

    # Use a global client if projects are same, but here target projects might differ.
    # The client needs credentials that can create transfers in the target project.
    # We'll instantiate one client and assume the user's ADC has permissions on all target projects.
    try:
        if not args.dry_run:
            client = bigquery_datatransfer.DataTransferServiceClient()
        else:
            client = None
    except Exception as e:
        logging.error(f"Failed to initialize DTS client: {e}")
        sys.exit(1)

    successful_configs = []
    
    # Generate timestamp once for all jobs in this run
    # Format: YYYYMMDDHHMMSS
    timestamp_prefix = datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')

    # Create all transfer configs first (without scheduling)
    for i, job in enumerate(jobs):
        config_name = create_transfer_config(
            client,
            job['source_project_id'],
            job['source_dataset_id'],
            job['target_project_id'],
            job['target_dataset_id'],
            job['service_account'],
            timestamp_prefix,
            dry_run=args.dry_run
        )
        
        if config_name:
            successful_configs.append(config_name)
        else:
            logging.error("Job creation failed. Stopping script.")
            sys.exit(1)

    logging.info(f"Successfully created/simulated {len(successful_configs)} transfer configs.")

    # Trigger manual runs for all created configs
    logging.info("Triggering manual runs for all configs...")
    
    current_time = datetime.now(timezone.utc)
    timestamp = timestamp_pb2.Timestamp()
    timestamp.FromDatetime(current_time)

    for config_name in successful_configs:
        if args.dry_run:
            logging.info(f"[DRY RUN] Would trigger manual run for config '{config_name}' with timestamp {current_time}")
            continue

        try:
            # The parent of the run is the transfer config name
            # requested_run_time is required for manual runs
            request = bigquery_datatransfer.StartManualTransferRunsRequest(
                parent=config_name,
                requested_run_time=timestamp
            )
            response = client.start_manual_transfer_runs(request=request)
            logging.info(f"Triggered manual run for config '{config_name}'")
        except Exception as e:
            logging.error(f"Failed to trigger manual run for '{config_name}': {e}")
            # We continue triggering others even if one fails
            
    logging.info("Finished processing all jobs.")

if __name__ == "__main__":
    main()
