#date: 2025-03-03T17:11:01Z
#url: https://api.github.com/gists/17fe4b1da52932485b1ab2cc62d63271
#owner: https://api.github.com/users/amdhing

import boto3
import time
import argparse
import socket
import statistics
import logging
from datetime import datetime, timezone

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Parse command-line arguments for regions
parser = argparse.ArgumentParser(description="DynamoDB replication test")
parser.add_argument("--write-region", required=True, help="AWS region to write to")
parser.add_argument("--read-region", required=True, help="AWS region to read from")
args = parser.parse_args()

WRITE_REGION = args.write_region
READ_REGION = args.read_region
TABLE_NAME = "mrsc_table"

# Initialize DynamoDB Clients
dynamodb_write = boto3.client("dynamodb", region_name=WRITE_REGION)
dynamodb_read = boto3.client("dynamodb", region_name=READ_REGION)

# Function to measure TCP ping latency
def measure_tcp_latency(region, count=20):
    host = f"dynamodb.{region}.amazonaws.com"
    port = 443  # HTTPS port
    logging.info(f"📡 Measuring TCP latency to {host}:{port} {count} times...")

    latencies = []
    for _ in range(count):
        try:
            start_time = time.time()
            with socket.create_connection((host, port), timeout=2):
                end_time = time.time()
                latency = (end_time - start_time) * 1000  # Convert to ms
                latencies.append(latency)
        except Exception as e:
            logging.error(f"⚠️ TCP ping error for {host}: {e}")

    avg_latency = statistics.mean(latencies) if latencies else None
    if avg_latency:
        logging.info(f"✅ Average TCP latency to {host}: {avg_latency:.2f} ms")
    else:
        logging.warning(f"⚠️ Could not measure TCP latency for {host}")

    return avg_latency

# Measure TCP latency for write and read regions
avg_tcp_latency_write = measure_tcp_latency(WRITE_REGION)
avg_tcp_latency_read = measure_tcp_latency(READ_REGION)

try:
    while True:
        # Generate new timestamped SK
        PK_VALUE = "Here"
        SK_VALUE = datetime.now(timezone.utc).isoformat()

        item = {
            "PK": {"S": PK_VALUE},
            "SK": {"S": SK_VALUE},
            "data": {"S": "This is a sample string."*100}
        }

        logging.info(f"\n📝 Writing item: PK={PK_VALUE}, SK={SK_VALUE} in {WRITE_REGION}...")

        # Measure write time
        start_time = time.time()
        dynamodb_write.put_item(TableName=TABLE_NAME, Item=item)
        end_time = time.time()
        write_latency = (end_time - start_time) * 1000  # Convert to milliseconds
        
        if avg_tcp_latency_write:
            logging.info(f"✅ Write completed in {write_latency:.2f} ms | Avg TCP RTT: {avg_tcp_latency_write:.2f} ms")
        else:
            logging.info(f"✅ Write completed in {write_latency:.2f} ms | Avg TCP RTT unavailable")

        # Measure replication time
        logging.info(f"🔍 Waiting for item to appear in {READ_REGION}...")
        start_time = time.time()
        counter = 0

        while True:
            response = dynamodb_read.get_item(
                TableName=TABLE_NAME,
                Key={"PK": {"S": PK_VALUE}, "SK": {"S": SK_VALUE}},
                ConsistentRead=True
            )
            counter += 1

            if "Item" in response:
                end_time = time.time()
                replication_latency = (end_time - start_time) * 1000  # Convert to milliseconds
                
                if avg_tcp_latency_read:
                    logging.info(f"✅ Item found in {READ_REGION} in attempt {counter} after {replication_latency:.2f} ms | Avg TCP RTT: {avg_tcp_latency_read:.2f} ms")
                else:
                    logging.info(f"✅ Item found in {READ_REGION} after {replication_latency:.2f} ms | (Avg TCP RTT unavailable)")
                break

            time.sleep(0.1)  # Poll every 100ms

        # Wait before next iteration
        time.sleep(2)

except KeyboardInterrupt:
    logging.info("\n🛑 Script stopped by user.")
