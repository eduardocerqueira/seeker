#date: 2025-07-08T16:59:54Z
#url: https://api.github.com/gists/2d99d40f54ac8283c6f2089a2ed73090
#owner: https://api.github.com/users/Sadhvik-Modak

#!/usr/bin/env python3
"""
Unity Catalog Table Access Auditor

This script audits Databricks Unity Catalog tables to check:
1. PAT token access to table metadata
2. SPN access to underlying ADLS Gen2 storage locations

* Requirements: 
pip3 install requests azure-identity azure-storage-file-datalake

* Usage

# Process all catalogs
python3 unity_catalog_access_auditor.py

# Process only the 'bronze' catalog
python3 unity_catalog_access_auditor.py --catalog bronze

# Test mode with specific catalog
python3 unity_catalog_access_auditor.py --catalog bronze --test

# Adjust performance parameters
python3 unity_catalog_access_auditor.py --catalog bronze --workers 20 --timeout 30
"""


import requests
import csv
from urllib.parse import urlparse
from azure.identity import ClientSecretCredential
from azure.storage.filedatalake import DataLakeServiceClient
import re
import concurrent.futures
import time
from datetime import datetime
import socket
import argparse
import sys

# Add a timeout for Azure operations to prevent long waits
AZURE_OPERATION_TIMEOUT = 15  # seconds

# Configuration settings
# Set to True for testing to exit after first catalog and first schema
TEST_MODE = False
# Set to True to use parallel processing for faster execution
PARALLEL_PROCESSING = True
MAX_WORKERS = 10  # Adjust based on your machine's capabilities

# Databricks connectivity info
DATABRICKS_HOST = "https://your-databricks-instance.azuredatabricks.net"  # Replace with your Databricks instance URL
DATABRICKS_TOKEN = "**********"

# SPN details for Azure access
CLIENT_ID = "your-client-id"  # Replace with your Azure SPN client ID
CLIENT_SECRET = "**********"
TENANT_ID = "your-tenant-id"  # Replace with your Azure tenant ID
SUBSCRIPTION_ID = "your-subscription-id"  # Replace with your Azure subscription ID


headers = {
    "Authorization": "**********"
}

# Add a debug mode flag to control verbose output
DEBUG_MODE = False  # Set to True to see API responses

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Databricks Table Audit Tool")
    parser.add_argument("--catalog", help="Process only the specified catalog")
    parser.add_argument("--skip-catalog-check", action="store_true", help="Skip catalog existence check (use when you don't have list catalog permissions)")
    parser.add_argument("--test", action="store_true", help="Run in test mode (process only first schema of first catalog)")
    parser.add_argument("--no-parallel", dest="parallel", action="store_false", help="Disable parallel processing")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help="Number of parallel workers")
    parser.add_argument("--timeout", type=int, default=AZURE_OPERATION_TIMEOUT, help="Timeout for Azure operations in seconds")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode to see API responses")
    return parser.parse_args()

def list_catalogs():
    url = f"{DATABRICKS_HOST}/api/2.1/unity-catalog/catalogs"
    print(f"📡 API Request: GET {url}")
    resp = requests.get(url, headers=headers)
    
    if DEBUG_MODE:
        print(f"📥 Response ({resp.status_code}): {resp.text[:500]}...")
    else:
        print(f"📥 Response Status: {resp.status_code}")
        
    return resp.json().get("catalogs", [])

def list_schemas(catalog):
    url = f"{DATABRICKS_HOST}/api/2.1/unity-catalog/schemas?catalog_name={catalog}"
    print(f"📡 API Request: GET {url}")
    resp = requests.get(url, headers=headers)
    
    if DEBUG_MODE:
        print(f"📥 Response ({resp.status_code}): {resp.text[:500]}...")
    else:
        print(f"📥 Response Status: {resp.status_code}")
        
    return resp.json().get("schemas", [])

def list_tables(catalog, schema):
    url = f"{DATABRICKS_HOST}/api/2.1/unity-catalog/tables?catalog_name={catalog}&schema_name={schema}"
    print(f"📡 API Request: GET {url}")
    resp = requests.get(url, headers=headers)
    
    if DEBUG_MODE:
        print(f"📥 Response ({resp.status_code}): {resp.text[:500]}...")
    else:
        print(f"📥 Response Status: {resp.status_code}")
        
    return resp.json().get("tables", [])

def describe_table(full_table_name):
    """Get table metadata using Databricks API with PAT token"""
    url = f"{DATABRICKS_HOST}/api/2.1/unity-catalog/tables/{full_table_name}"
    print(f"📡 API Request: GET {url}")
    resp = requests.get(url, headers=headers)
    
    if DEBUG_MODE:
        print(f"📥 Response ({resp.status_code}): {resp.text[:500]}...")
    
    # Return both the status code and the response for better handling
    if resp.status_code == 200:
        return {"status": "success", "data": resp.json()}
    elif resp.status_code in [401, 403]:
        return {"status": "access_denied", "data": {}, "error": f"PAT access denied (HTTP {resp.status_code})"}
    else:
        return {"status": "error", "data": {}, "error": f"HTTP {resp.status_code}: {resp.text[:100]}"}

def save_to_csv(rows, filename="databricks_table_inventory.csv"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"databricks_table_inventory_{timestamp}.csv"
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "catalog.schema.table", 
            "catalog_name", 
            "schema_name", 
            "table_name", 
            "storage_location", 
            "table_type", 
            "pat_access", 
            "spn_access", 
            "container", 
            "account", 
            "path", 
            "client_id"
        ])
        writer.writerows(rows)
    print(f"✅ CSV written: {filename}")

def parse_storage_location(storage_location):
    if not storage_location or storage_location == "N/A":
        return None, None, None
    
    if not storage_location.startswith("abfss://"):
        return None, None, None
        
    try:
        parsed = urlparse(storage_location)
        container = parsed.netloc.split('@')[0]
        account_name = parsed.netloc.split('@')[1].split('.')[0]
        path = parsed.path.lstrip('/')
        return container, account_name, path
    except:
        return None, None, None

def test_access(storage_location):
    try:
        if not storage_location or storage_location == "N/A":
            return "No (Invalid Location)"

        # Only handle ADLS Gen2 locations
        if not storage_location.startswith("abfss://"):
            return "No (Not ADLS)"
            
        # Parse the ADLS Gen2 path
        container, account_name, path = parse_storage_location(storage_location)
        if not container or not account_name:
            return "No (Invalid ADLS Format)"
        
        # Set a default socket timeout to prevent hanging on network operations
        original_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(AZURE_OPERATION_TIMEOUT)
        
        try:
            # Get token credential using the provided SPN details
            credential = "**********"
                tenant_id=TENANT_ID,
                client_id=CLIENT_ID,
                client_secret= "**********"
            )
            
            # Create a DataLake service client
            service_client = DataLakeServiceClient(
                account_url=f"https://{account_name}.dfs.core.windows.net",
                credential=credential
            )
            
            # Get a filesystem (container) client
            file_system_client = service_client.get_file_system_client(container)
            
            # Test if container exists
            try:
                container_exists = file_system_client.exists()
                if not container_exists:
                    return "No (Container Not Found)"
            except Exception as e:
                return f"No (Container Access Error)"
                
            # If path is specified, check if the directory/file exists
            if path:
                try:
                    # Quick existence check is faster than trying to list files
                    directory_client = file_system_client.get_directory_client(path)
                    if directory_client.exists():
                        return "Yes"
                    
                    # If directory doesn't exist, try as a file
                    last_slash = path.rfind('/')
                    if last_slash >= 0:
                        parent_dir = path[:last_slash]
                        file_name = path[last_slash+1:]
                        
                        # Check if parent directory exists
                        parent_dir_client = file_system_client.get_directory_client(parent_dir)
                        if not parent_dir_client.exists():
                            return "No (Parent Directory Not Found)"
                            
                        file_client = parent_dir_client.get_file_client(file_name)
                        if file_client.exists():
                            return "Yes"
                    else:
                        file_client = file_system_client.get_file_client(path)
                        if file_client.exists():
                            return "Yes"
                    
                    return "No (Path Not Found)"
                except socket.timeout:
                    return "No (Operation Timeout)"
                except Exception:
                    return "No (Access Error)"
            else:
                # Just check container access
                return "Yes (Container Only)"
        finally:
            # Restore original socket timeout
            socket.setdefaulttimeout(original_timeout)
    except socket.timeout:
        return "No (Operation Timeout)"
    except Exception:
        return "No (Error)"

def process_table(catalog_name, schema_name, table):
    """Process a single table and return its data"""
    table_name = table['name']
    full_table = f"{catalog_name}.{schema_name}.{table_name}"
    print(f"        🔸 Processing: {full_table}")
    
    # Check PAT access first
    metadata_response = describe_table(table["full_name"])
    pat_access = "Yes" if metadata_response["status"] == "success" else "No"
    
    # Get metadata if PAT access was successful
    metadata = metadata_response.get("data", {})
    storage_location = metadata.get("storage_location", "N/A")
    table_type = metadata.get("table_type", "UNKNOWN")
    
    # Additional details about PAT access
    if metadata_response["status"] != "success":
        pat_access = f"No ({metadata_response.get('error', 'Unknown error')})"
    
    container, account, path = parse_storage_location(storage_location)
    
    # Only test SPN access if we have a valid storage location
    spn_access = "N/A"
    if storage_location != "N/A" and pat_access.startswith("Yes"):
        start_time = time.time()
        # Set a timeout for the overall operation
        spn_access = test_access(storage_location)
        end_time = time.time()
        elapsed = end_time - start_time
        
        # If it took longer than expected, note that in the output
        time_note = f"({elapsed:.2f}s)"
        if elapsed > AZURE_OPERATION_TIMEOUT:
            time_note = f"({elapsed:.2f}s - SLOW)"
            
        print(f"        🔸 Table: {full_table} - PAT: {pat_access}, SPN: {spn_access} {time_note}")
    else:
        print(f"        🔸 Table: {full_table} - PAT: {pat_access}, SPN: {spn_access} (skipped SPN check)")
    
    # Added separate PAT and SPN access columns
    return [
        full_table, 
        catalog_name, 
        schema_name, 
        table_name, 
        storage_location, 
        table_type, 
        pat_access, 
        spn_access, 
        container, 
        account, 
        path, 
        CLIENT_ID
    ]

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Override global settings with command line arguments
    global TEST_MODE, PARALLEL_PROCESSING, MAX_WORKERS, AZURE_OPERATION_TIMEOUT, DEBUG_MODE
    if args.test:
        TEST_MODE = True
    if args.parallel is not None:
        PARALLEL_PROCESSING = args.parallel
    if args.workers:
        MAX_WORKERS = args.workers
    if args.timeout:
        AZURE_OPERATION_TIMEOUT = args.timeout
    if args.debug:
        DEBUG_MODE = True
        print("🐛 Debug mode enabled - API responses will be printed")

    target_catalog = args.catalog
    skip_catalog_check = args.skip_catalog_check
    
    start_time_total = time.time()
    output_rows = []
    
    # Handle direct catalog access (when user doesn't have list catalog permissions)
    if target_catalog and skip_catalog_check:
        print(f"🔍 Directly accessing catalog: {target_catalog} (skipping catalog existence check)")
        
        try:
            # Try to access the schemas in the specified catalog
            schemas = list_schemas(target_catalog)
            
            if not schemas:
                print(f"⚠️ No schemas found in catalog '{target_catalog}' or no access to this catalog.")
                return
                
            print(f"📁 Catalog: {target_catalog}")
            print(f"  📚 Found {len(schemas)} schemas")
            
            process_catalog_schemas(target_catalog, schemas, output_rows)
            
        except Exception as e:
            print(f"❌ Error accessing catalog '{target_catalog}': {str(e)}")
            sys.exit(1)
    
    # Normal flow - list available catalogs first
    else:
        try:
            all_catalogs = list_catalogs()
            
            # Filter catalogs if a specific one was requested
            catalogs = []
            if target_catalog:
                for catalog in all_catalogs:
                    if catalog["name"] == target_catalog:
                        catalogs = [catalog]
                        break
                if not catalogs:
                    print(f"❌ Error: Catalog '{target_catalog}' not found. Available catalogs: {', '.join([c['name'] for c in all_catalogs])}")
                    sys.exit(1)
            else:
                catalogs = all_catalogs
            
            print(f"🔍 Found {len(all_catalogs)} catalogs, processing {len(catalogs)}")
            
            if not catalogs:
                print("No catalogs found.")
                return
            
            for i, catalog in enumerate(catalogs):
                catalog_name = catalog["name"]
                print(f"\n📁 Catalog: {catalog_name} ({i+1}/{len(catalogs)})")
                schemas = list_schemas(catalog_name)
                print(f"  📚 Found {len(schemas)} schemas")
                
                process_catalog_schemas(catalog_name, schemas, output_rows)
                
                # Exit after processing the first catalog if in test mode
                if TEST_MODE and i == 0:
                    print(f"\n⚠️ TEST MODE: Exiting after first catalog ({catalog_name}). Use --no-test to process all catalogs.")
                    break
                
        except Exception as e:
            print(f"❌ Error listing catalogs: {str(e)}")
            print("If you don't have permissions to list catalogs, try using --catalog with --skip-catalog-check")
            sys.exit(1)
    
    end_time_total = time.time()
    print(f"\n✅ Total execution time: {(end_time_total-start_time_total):.2f} seconds")
    print(f"✅ Processed {len(output_rows)} tables")
    
    save_to_csv(output_rows)

def process_catalog_schemas(catalog_name, schemas, output_rows):
    """Process all schemas in a catalog"""
    for j, schema in enumerate(schemas):
        schema_name = schema["name"]
        print(f"    📂 Schema: {schema_name} ({j+1}/{len(schemas)})")
        tables = list_tables(catalog_name, schema_name)
        print(f"      🧾 Found {len(tables)} tables")

        if PARALLEL_PROCESSING and tables:
            # Process tables in parallel for better performance
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                # Create a list of future tasks
                future_to_table = {
                    executor.submit(process_table, catalog_name, schema_name, table): table 
                    for table in tables
                }
                
                # Process completed tasks as they complete
                for future in concurrent.futures.as_completed(future_to_table):
                    result = future.result()
                    output_rows.append(result)
        else:
            # Process tables sequentially
            for table in tables:
                result = process_table(catalog_name, schema_name, table)
                output_rows.append(result)
        
        # Exit after processing the first schema if in test mode
        if TEST_MODE and j == 0:
            print(f"\n⚠️ TEST MODE: Exiting after first schema ({schema_name}). Use --no-test to process all schemas.")
            break

if __name__ == "__main__":
    main()
after first schema ({schema_name}). Use --no-test to process all schemas.")
            break

if __name__ == "__main__":
    main()
