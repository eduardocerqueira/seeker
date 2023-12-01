#date: 2023-12-01T16:55:05Z
#url: https://api.github.com/gists/b02d38d05c6669600e3c6ca279502434
#owner: https://api.github.com/users/davidallan

import oci
import json
import requests
import sys
import glob
import os
from oci.signer import Signer

workspaceID = sys.argv [1]
applicationKey = sys.argv[2]
taskKey = sys.argv[3]
taskRunKey = sys.argv[4]
region = sys.argv[5]

config = oci.config.from_file('~/.oci/config')

data_integration_client = oci.data_integration.DataIntegrationClient(config=config)
md = oci.data_integration.models.RegistryMetadata(aggregator_key=taskKey)
task = oci.data_integration.models.CreateTaskRunDetails(registry_metadata=md, ref_task_run_id=taskRunKey,re_run_type=oci.data_integration.models.CreateTaskRunDetails.RE_RUN_TYPE_BEGINNING)
tskrun = data_integration_client.create_task_run(workspaceID,applicationKey, create_task_run_details=task)
print('Task Run ' + tskrun.data.key)