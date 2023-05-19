#date: 2023-05-19T16:53:46Z
#url: https://api.github.com/gists/8ff2ce1c763af2e3429024d2a4d30d9f
#owner: https://api.github.com/users/ernestoongaro

import enum
import os
import time

# Be sure to `pip install requests` in your python environment
import requests

ACCOUNT_ID = 39
JOB_ID = 302

# Store your dbt Cloud API token securely in your workflow tool
API_KEY = '<put in your dbt Cloud API key here'


def _trigger_job() -> int:
    res = requests.post(
        url=f"https://emea.dbt.com/api/v2/accounts/{ACCOUNT_ID}/jobs/{JOB_ID}/run/",
        headers={'Authorization': "**********"
        json={
            'cause': f"Triggered by Fivetran",
        }
    )

    response_payload = res.json()
    return response_payload['data']['id']



def run(request):
    job_run_id = _trigger_job()

    return(f"job_run_id = {job_run_id}")

  


)

  


