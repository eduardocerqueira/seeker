#date: 2024-01-09T16:48:07Z
#url: https://api.github.com/gists/3e16fce42a9e7655d3d49257412656f8
#owner: https://api.github.com/users/myersCody

# You can update the make command to run this script:
# shell-schema: schema := org1234567
# shell-schema:
# 	$(DJANGO_MANAGE) tenant_command shell --schema=$(schema) < create_reindex_parquet_files.py

import os
import pandas as pd
from masu.util.common import get_path_prefix
from api.models import Provider
from dateutil import parser
from api.utils import DateHelper
from masu.api.upgrade_trino.util.task_handler import FixParquetTaskHandler
import boto3


PANDA_KWARGS = {
            "allow_truncated_timestamps": True,
            "coerce_timestamps": "ms",
            "index": False,
        }
PARQUET_DATA_TYPE = "parquet"
SCHEMA = "org1234567"
S3_BUCKET_NAME = "koku-bucket"
S3_ENDPOINT = "http://localhost:9000"
S3_ACCESS_KEY = "**********"
S3_SECRET = "**********"



def create_parquet_file_with_reindex(provider):
    """Creates a parquet file using reindexing."""
    data_frame = pd.DataFrame()
    data_frame = data_frame.reindex(columns=FixParquetTaskHandler.clean_column_names(provider.type))
    filename = f"test_{str(provider.uuid)}.{PARQUET_DATA_TYPE}"
    data_frame.to_parquet(filename, **PANDA_KWARGS)
    return filename


def _parquet_path_s3(bill_date, report_type, provider_type, provider_uuid):
    """The path in the S3 bucket where Parquet files are loaded."""
    return get_path_prefix(
        SCHEMA,
        provider_type,
        provider_uuid,
        bill_date,
        PARQUET_DATA_TYPE,
        report_type=report_type,
    )

def _parquet_daily_path_s3(bill_date, report_type, provider_type, provider_uuid):
    """The path in the S3 bucket where Parquet files are loaded."""
    if report_type is None:
        report_type = "raw"
    return get_path_prefix(
        SCHEMA,
        provider_type,
        provider_uuid,
        bill_date,
        PARQUET_DATA_TYPE,
        report_type=report_type,
        daily=True,
    )

def _parquet_ocp_on_cloud_path_s3(bill_date, provider_type, provider_uuid):
    """The path in the S3 bucket where Parquet files are loaded."""
    return get_path_prefix(
        SCHEMA,
        provider_type,
        provider_uuid,
        bill_date,
        PARQUET_DATA_TYPE,
        report_type="openshift",
        daily=True,
    )

def report_types(provider_type):
    if provider_type == Provider.PROVIDER_OCI:
        return ["cost", "usage"]
    if provider_type == Provider.PROVIDER_OCP:
        return ["namespace_labels", "node_labels", "pod_usage", "storage_usage"]
    return [None]


providers = Provider.objects.filter(active=True, paused=False)
start_date = parser.parse("2023-11-01").replace(day=1)
dh = DateHelper()
bill_datetimes = dh.list_months(start_date, dh.today.replace(tzinfo=None))
for bill_date in bill_datetimes:
    for provider in providers:
        provider_type = provider.type
        provider_type = provider_type.replace("-local", "")
        if provider_type in [Provider.PROVIDER_GCP, Provider.PROVIDER_IBM]:
            continue
        for report_type in report_types(provider.type):
            parquet_paths = [
                _parquet_path_s3(bill_date, report_type, provider_type, str(provider.uuid)),
                _parquet_daily_path_s3(bill_date, report_type, provider_type, str(provider.uuid)),
            ]
            if provider_type in [Provider.PROVIDER_AWS, Provider.PROVIDER_AZURE]:
                parquet_paths.append(_parquet_ocp_on_cloud_path_s3(bill_date, provider_type, str(provider.uuid)))
            file_name = create_parquet_file_with_reindex(provider)
            aws_session = boto3.Session(
                    aws_access_key_id= "**********"
                    aws_secret_access_key= "**********"
                )
            for s3_path in parquet_paths:
                s3_resource = aws_session.resource("s3", endpoint_url=S3_ENDPOINT)
                s3_full_path = s3_path + "/" + file_name

                s3_obj = {"bucket_name": S3_BUCKET_NAME, "key": s3_full_path}
                upload = s3_resource.Object(**s3_obj)
                with open(file_name, "rb") as fin:
                    upload.upload_fileobj(fin)
            os.remove(file_name)

_name)

