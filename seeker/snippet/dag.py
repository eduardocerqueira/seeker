#date: 2025-03-31T17:02:14Z
#url: https://api.github.com/gists/d9a39816c310de5788ef180b27d53ce0
#owner: https://api.github.com/users/duboc

import google.auth.transport.requests
import google.oauth2.id_token
import requests
from airflow.operators.python import PythonOperator
from airflow.models.dag import DAG
from datetime import datetime

# Replace with your Cloud Function URL
FUNCTION_URL = "https_your_function_trigger_url"

def call_cloud_function():
    """Calls a secured Cloud Function using ADC."""

    # ADC automatically finds the Composer environment's service account
    auth_req = google.auth.transport.requests.Request()
    id_token = "**********"

    headers = {"Authorization": "**********"

    try:
        response = requests.post(FUNCTION_URL, headers=headers, timeout=60) # Or GET, etc.
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        print(f"Function Response Status: {response.status_code}")
        print(f"Function Response Text: {response.text}")
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error calling Cloud Function: {e}")
        raise

with DAG(
    dag_id='call_cloud_function_with_adc',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:
    call_function_task = PythonOperator(
        task_id='call_the_cloud_function',
        python_callable=call_cloud_function,
    )ction',
        python_callable=call_cloud_function,
    )