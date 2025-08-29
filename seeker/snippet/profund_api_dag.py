#date: 2025-08-29T17:00:20Z
#url: https://api.github.com/gists/796e006a41f90247e5381b24448a9174
#owner: https://api.github.com/users/mtaft904

import datetime
from airflow import DAG
from etsy.utils.defaults import patch_all

from airflow.providers.google.cloud.hooks.gcs import GCSHook
from airflow.providers.google.cloud.transfers.gcs_to_bigquery import (
    GCSToBigQueryOperator,
)
from airflow.decorators import task
import logging
import json
import requests

patch_all()

DEFAULT_TASK_ARGS = {
    "owner": "seo-martech",
    "retries": 1,
    "retry_delay": datetime.timedelta(seconds=300),
    "start_date": datetime.datetime(2025, 8, 1, 0, 0),
}

OUTPUT_BUCKET = "etsy-sitemaps-prod-raw-data"
OUTPUT_PATH = "profound/{{ ds }}/visibility.json"

# separate this task into its own file under /include/extras/
@task
def profound_api_to_gcs(api_key: str):
    data = {
        "date_interval": "day",
        "dimensions": [
            "date",
            "asset_name",
            "model",
            "topic",
            "tag"
        ],
        "filters": [],
        "order_by": {},
        "pagination": {
            "offset": 0
        },
        "category_id": "7786a90d-f152-415f-a892-45286eb7fb0c",
        "start_date": "2025-08-28",
        "end_date": "2025-08-29",
        "metrics": [
            "mentions_count",
            "share_of_voice",
            "visibility_score"
        ]
    }
    
    r = requests.post("https://api.tryprofound.com/v1/reports/visibility?{{ api_key }}", json=data)
    logging.info("Profound API response status: %s", r.status_code)

    results = r.json()

    data = results["data"]
    info = results["info"]
    metrics_schema = info["query"]["metrics"]
    dimensions_schema = info["query"]["dimensions"]

    formatted_results = list()
    for item in data:
        details = dict()
        for index, value in enumerate(dimensions_schema):
            details[value] = item["dimensions"][index]
    
        for index, value in enumerate(metrics_schema):
            details[value] = item["metrics"][index]

        formatted_results.append(details)

    logging.info("Uploading results to GCS")

    hook = GCSHook()
    
    # make sure data is formatted correctly for bigquery
    hook.upload(
        bucket_name=OUTPUT_BUCKET,
        object_name=OUTPUT_PATH,
        data=json.dumps(formatted_results),
        mime_type="application/json",
    )

with DAG(
    max_active_runs=1,
    tags=["seo"],
    catchup=False,
    dag_id="seo_profound_visibility",
    schedule="@daily",
    default_args=DEFAULT_TASK_ARGS,
) as dag: 
    
    # fetch api key
    
    profound_api_to_gcs = profound_api_to_gcs(api_key)

    upload_to_bigquery = GCSToBigQueryOperator(
        source_format="NEWLINE_DELIMITED_JSON",
        destination_project_dataset_table='etsy-data-warehouse-prod.seo.profound_visibility',
        task_id="upload_to_bigquery",
        write_disposition="WRITE_APPEND",
        time_partitioning={"type": "DAY"},
        source_objects=[OUTPUT_PATH],
        bucket=OUTPUT_BUCKET,
        create_disposition="CREATE_IF_NEEDED",
        schema_update_options=["ALLOW_FIELD_ADDITION"],
        schema_fields=[
            {"name": "topic", "type": "STRING", "mode": "NULLABLE"},
            {"name": "model", "type": "STRING", "mode": "NULLABLE"},
            {"name": "date", "type": "STRING", "mode": "NULLABLE"},
            {"name": "asset_name", "type": "STRING", "mode": "NULLABLE"},
            {"name": "tag", "type": "STRING", "mode": "NULLABLE"},
            {"name": "visibility_score", "type": "FLOAT", "mode": "NULLABLE"},
            {"name": "mentions_count", "type": "FLOAT", "mode": "NULLABLE"},
            {"name": "share_of_voice", "type": "FLOAT", "mode": "NULLABLE"},
        ],
    )

    profound_api_to_gcs >> upload_to_bigquery