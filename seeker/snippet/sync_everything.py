#date: 2025-09-04T16:51:37Z
#url: https://api.github.com/gists/6ed04f79fccc02ee0755921c2fdfea0d
#owner: https://api.github.com/users/MarkPryceMaherMSFT

import requests
from notebookutils import mssparkutils
import sempy.fabric as fabric 
from sempy.fabric.exceptions import FabricHTTPException, WorkspaceNotFoundException 
import json
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType
from datetime import datetime

# Get token
token = "**********"

# Workspace ID
workspace_id=spark.conf.get("trident.workspace.id")
workspace_name = notebookutils.mssparkutils.env.getWorkspaceName()
# API endpoint
url = f"https://api.fabric.microsoft.com/v1/workspaces/{workspace_id}/lakehouses"

# Headers
headers = {
    "Authorization": "**********"
    "Content-Type": "application/json"
}


# Define schema for sync status table
schema = StructType([
    StructField("syncRunDateTime", StringType(), True),  # New field
    StructField("workspace_name", StringType(), True),
    StructField("workspace_id", StringType(), True),
    StructField("lakehouse_name", StringType(), True),
    StructField("lakehouse_id", StringType(), True),
    StructField("sql_endpoint_id", StringType(), True),
    StructField("sync_status", StringType(), True),
    StructField("tableName", StringType(), True),
    StructField("status", StringType(), True),
    StructField("startDateTime", StringType(), True),
    StructField("endDateTime", StringType(), True),
    StructField("lastSuccessfulSyncDateTime", StringType(), True)
])


# Collect sync status records
records = []

#Instantiate the client
client = fabric.FabricRestClient()

# Make the request
response = requests.get(url, headers=headers)
lakehouses = response.json()["value"]

# Loop through lakehouses
for lh in lakehouses:

    name = lh.get("displayName")
    lakehouse_id = lh.get("id")
    sql_props = lh.get("properties", {}).get("sqlEndpointProperties", {})
    sql_endpoint_id = sql_props.get("id")

    print(f"Lakehouse Name: {name}")
    print(f"Lakehouse ID: {lakehouse_id}")
    print(f"SQL Endpoint ID: {sql_endpoint_id}")
    print("-" * 40)

    
    # This is the SQL endpoint I want to sync with the lakehouse, this needs to be the GUI
    sqlendpoint = lh['properties']['sqlEndpointProperties']['id']

    # URI for the call 
    uri = f"v1/workspaces/{workspace_id}/sqlEndpoints/{sqlendpoint}/refreshMetadata" 

    # Example of setting a timeout
    payload = { "timeout": {"timeUnit": "Seconds", "value": "60"}  }  

    try:
        sync_run_time = datetime.utcnow().isoformat() + "Z"
        response = client.post(uri,json= payload, lro_wait = True) 
        sync_status = json.loads(response.text)
        #display(sync_status)
    
        records = [
            (
                sync_run_time,
                workspace_name,
                workspace_id,
                name,
                lakehouse_id,
                sql_props,
                sql_endpoint_id,
                item['tableName'],
                item['status'],
                item['startDateTime'],
                item['endDateTime'],
                item['lastSuccessfulSyncDateTime']
            )
            for item in sync_status['value']
        ]   

        df = spark.createDataFrame(records, schema=schema)
        df.write.mode("append").saveAsTable("lakehouse_sync_status")
    except Exception as e: print(e)
  except Exception as e: print(e)
