#date: 2022-05-04T17:06:43Z
#url: https://api.github.com/gists/bb8fd618743c3d12b70bf801fe2e82e8
#owner: https://api.github.com/users/Intelrunner

""" Warning - this script automatically submits a request for GCP DLP job creation
to scan for every table, in every dataset available to a user. This will fail if the following APIs / Permissions
are not enabled. 

This script can, and may, cost you actual $$. Outcomes can be seen in the DLP console.

DLP - jobs.create, jobs.get, jobs.list

BQ - bigquery.user (role)

"""
# Import the BQ
from google.cloud import bigquery
# Import the client library
from google.cloud import dlp_v2

# Instantiate a client.
dlp_client = dlp_v2.DlpServiceClient()


# Construct a BigQuery client object.
client = bigquery.Client()

datasets = list(client.list_datasets())  # Make an API request.
project = client.project

dataset_list = []

if datasets:
    print("Datasets in project {}:".format(project))
    for dataset in datasets:
        # ("\t{}".format(dataset.dataset_id))
        dataset_list.append(dataset.dataset_id)
    print(dataset_list)
else:
    print("{} project does not contain any datasets.".format(project))


for dataset in dataset_list:
    table_list = []
    tables = client.list_tables(dataset)  # Make an API request.
    print("Tables contained in '{}':".format(dataset))

    for table in tables:
        table_uri = ("{}.{}.{}".format(
            table.project, table.dataset_id, table.table_id))
        print(table_uri)

        inspect_job_data = {
            'storage_config': {
                'big_query_options': {
                    'table_reference': {
                        'project_id': table.project,
                        'dataset_id': table.dataset_id,
                        'table_id': table.table_id
                    },
                    'rows_limit': 10000,
                    'sample_method': 'RANDOM_START',
                },
            },
            'inspect_config': {
                'info_types': [
                    {'name': 'ALL_BASIC'},
                ],
            },
            'actions': [
                {
                    'save_findings': {
                        'output_config': {
                            'table': {
                                'project_id': table.project,
                                'dataset_id': table.dataset_id,
                                'table_id': '{}_DLP'.format(table.table_id)
                            }
                        }

                    },
                },
            ]
        }
        parent = "project/{}".format(table.project)
        operation = dlp_client.create_dlp_job(
            parent=parent, inspect_job=inspect_job_data)