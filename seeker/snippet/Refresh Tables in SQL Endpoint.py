#date: 2026-03-03T17:22:29Z
#url: https://api.github.com/gists/573fad0cc5dc3dd705e65976aa00d575
#owner: https://api.github.com/users/zan-tavcar

########################################################################################################
# Sample script to call the syncronisation between the Fabric Lakehouse and the SQL Endpoint
#        
## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
#  This script is a workaround until the documented API is released: https://learn.microsoft.com/en-us/fabric/release-plan/data-warehouse#refresh-sql-analytics-endpoint-rest-api
#
#sempy version 0.4.0 or higher
!pip install semantic-link --q 
import json
import time
import struct
import sqlalchemy
import pyodbc
import notebookutils
import pandas as pd
from pyspark.sql import functions as fn
from datetime import datetime
import sempy.fabric as fabric
from sempy.fabric.exceptions import FabricHTTPException, WorkspaceNotFoundException

def pad_or_truncate_string(input_string, length, pad_char=' '):
    # Truncate if the string is longer than the specified length
    if len(input_string) > length:
        return input_string[:length]
    # Pad if the string is shorter than the specified length
    return input_string.ljust(length, pad_char)

## not needed, but usefull
tenant_id=spark.conf.get("trident.tenant.id")
workspace_id=spark.conf.get("trident.workspace.id")
lakehouse_id=spark.conf.get("trident.lakehouse.id")
lakehouse_name=spark.conf.get("trident.lakehouse.name")
#sql_endpoint= fabric.FabricRestClient().get(f"/v1/workspaces/{workspace_id}/lakehouses/{lakehouse_id}").json()['properties']['sqlEndpointProperties']['connectionString']

#Instantiate the client
client = fabric.FabricRestClient()

# This is the SQL endpoint I want to sync with the lakehouse, this needs to be the GUI
sqlendpoint = fabric.FabricRestClient().get(f"/v1/workspaces/{workspace_id}/lakehouses/{lakehouse_id}").json()['properties']['sqlEndpointProperties']['id']
j_son = fabric.FabricRestClient().get(f"/v1/workspaces/{workspace_id}/lakehouses/{lakehouse_id}").json()
#display(j_son)

# URI for the call
uri = f"/v1.0/myorg/lhdatamarts/{sqlendpoint}"
# This is the action, we want to take
payload = {"commands":[{"$type":"MetadataRefreshExternalCommand"}]}

# code for testing 
# Test 1 : test creating a new table
#spark.sql("drop table if exists test1")
#spark.sql("create table test1 as SELECT * FROM lakehouse.Date LIMIT 1000")
# Test 2 : create a duplicate of the table with a different case
#df = spark.sql("SELECT * FROM lakehouse.Date LIMIT 1000")
#df.write.save("Tables/Test1")

# Call the REST API
response = client.post(uri,json= payload)
## You should add some error handling here

# return the response from json into an object we can get values from
data = json.loads(response.text)

# We just need this, we pass this to call to check the status
batchId = data["batchId"]

# the state of the sync i.e. inProgress
progressState = data["progressState"]

# URL so we can get the status of the sync
statusuri = f"/v1.0/myorg/lhdatamarts/{sqlendpoint}/batches/{batchId}"

statusresponsedata = ""

while progressState == 'inProgress' :
    # For the demo, I have removed the 1 second sleep.
    time.sleep(1)

    # check to see if its sync'ed
    #statusresponse = client.get(statusuri)

    # turn response into object
    statusresponsedata = client.get(statusuri).json()

    # get the status of the check
    progressState = statusresponsedata["progressState"]
    # show the status
    display(f"Sync state: {progressState}")

# if its good, then create a temp results, with just the info we care about
if progressState == 'success':
    table_details = [
        {
          'tableName': table['tableName'],
         'warningMessages': table.get('warningMessages', []),
         'lastSuccessfulUpdate': table.get('lastSuccessfulUpdate', 'N/A'),
         'tableSyncState':  table['tableSyncState'],
         'sqlSyncState':  table['sqlSyncState']
        }
        for table in statusresponsedata['operationInformation'][0]['progressDetail']['tablesSyncStatus']
    ]

# if its good, then shows the tables
if progressState == 'success':
    # Print the extracted details
    print("Extracted Table Details:")
    for detail in table_details:
        print(f"Table: {pad_or_truncate_string(detail['tableName'],30)}   Last Update: {detail['lastSuccessfulUpdate']}  tableSyncState: {detail['tableSyncState']}   Warnings: {detail['warningMessages']}")

## if there is a problem, show all the errors
if progressState == 'failure':
    # display error if there is an error
    display(statusresponsedata)
    