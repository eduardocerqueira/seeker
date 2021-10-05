#date: 2021-10-05T17:10:43Z
#url: https://api.github.com/gists/0271d2192752cd46fb3e9c8e266bdcd0
#owner: https://api.github.com/users/steffen-workpath

import logging
import requests


WORKPATH_API_CLIENT = "d5495135-190a-4c50-9890-0417e251f470"
WORKPATH_API_BASE = "https://connect.workpath.com/api/v2"


def load_kpis_from_bi():
    # Replace this with code to get relevant KPIs from your BI, and map them to the Workpath KPI IDs. Make a call to
    # https://connect.workpath.com/api/v2/kpis to list all current KPIs including their IDs.
    return [
        {
            "workpath_id": "78af42",
            "value": 567_055.54,
        },
        # ...
    ]


def update_kpi_in_workpath(workpath_id, value):
    # Make request to Workpath API
    url = f"{WORKPATH_API_BASE}/kpis/{workpath_id}"
    json_data = {
        "current_value": value
    }
    headers = {
        "Authorization": f"Bearer {WORKPATH_API_CLIENT}"
    }
    response = requests.patch(url, json=json_data, headers=headers)

    # Error handling
    if response.status_code != 200:
        logging.error(f"Request to {url} failed. Error code: {response.status_code}. JSON body: {json_data}")


if __name__ == "__main__":

    for kpi in load_kpis_from_bi():
        update_kpi_in_workpath(kpi["workpath_id"], kpi["value"])
