#date: 2024-03-19T17:06:33Z
#url: https://api.github.com/gists/a2d7973ab43506b74b7275fde87ead9a
#owner: https://api.github.com/users/countmapula

# https://github.com/microsoft/azure-devops-python-samples/blob/main/src/samples/work_item_tracking.py
# https://medium.com/@rohit8450/create-a-new-work-item-in-azure-devops-using-python-api-d7f33f95c3c4
# https://stackoverflow.com/questions/57001917/how-i-can-get-attachments-detail-for-my-workitem-using-azure-devops-rest-api
# https://stackoverflow.com/questions/66642121/azure-rest-api-how-to-download-an-attachment-from-the-workitem-python

"""
WIT samples
"""
import datetime
import json 
import logging

from azure.devops.connection import Connection
from datetime import datetime
from msrest.authentication import BasicAuthentication
from azure.devops.v7_1.work_item_tracking.models import JsonPatchOperation, Wiql

run_at = datetime.now().strftime("%Y%m%d%H%M%S")

# Your Azure DevOps organization URL and personal access token
organization_url = "https://dev.azure.com/myorg/"
personal_access_token = "**********"
project_name = "myproject"
work_item_type = "Task"

# Create a connection to Azure DevOps
credentials = "**********"
connection = Connection(base_url=organization_url, creds=credentials)
wit_client = connection.clients.get_work_item_tracking_client()


# ˙from samples import resource
# from utils import emit

# logger = logging.getLogger(__name__)


desired_ids = "1"


work_items = wit_client.get_work_items(ids=desired_ids, error_policy="omit", expand="All")

for item in work_items:

    print(item)
    if len(item.relations) > 0:
            for rel in item.relations:
                print(rel)

specific_item = wit_client.get_work_item(1, expand="All")

print(specific_item)

def print_work_item(work_item):
    print(
        "{0} {1}: {2}".format(
            work_item.fields["System.WorkItemType"],
            work_item.id,
            work_item.fields["System.Title"],
        )
    )


# @resource("work_items")
def get_work_items(context):
    wit_client = context.connection.clients.get_work_item_tracking_client()

    desired_ids = range(1, 51)
    work_items = wit_client.get_work_items(ids=desired_ids, error_policy="omit", expand="All")

    for id_, work_item in zip(desired_ids, work_items):
        if work_item:
            print_work_item(work_item)
        else:
            print("(work item {0} omitted by server)".format(id_))

    return work_items


# @resource("work_items")
def get_work_items_as_of(context):
    wit_client = context.connection.clients.get_work_item_tracking_client()

    desired_ids = range(1, 51)
    as_of_date = datetime.datetime.now() + datetime.timedelta(days=-7)
    work_items = wit_client.get_work_items(
        ids=desired_ids, as_of=as_of_date, error_policy="omit"
    )

    for id_, work_item in zip(desired_ids, work_items):
        if work_item:
            print_work_item(work_item)
        else:
            print("(work item {0} omitted by server)".format(id_))

    return work_items


# @resource("wiql_query")
def wiql_query(context):
    wit_client = context.connection.clients.get_work_item_tracking_client()
    wiql = Wiql(
        query="""
        select [System.Id],
            [System.WorkItemType],
            [System.Title],
            [System.State],
            [System.AreaPath],
            [System.IterationPath],
            [System.Tags]
        from WorkItems
        where [System.WorkItemType] = "Test Case"
        order by [System.ChangedDate] desc"""
    )
    # We limit number of results to 30 on purpose
    wiql_results = wit_client.query_by_wiql(wiql, top=30).work_items
    print("Results: {0}".format(len(wiql_results)))
    if wiql_results:
        # WIQL query gives a WorkItemReference with ID only
        # => we get the corresponding WorkItem from id
        work_items = (
            wit_client.get_work_item(int(res.id)) for res in wiql_results
        )
        for work_item in work_items:
            print_work_item(work_item)
        return work_items
    else:
        return []    else:
        return []