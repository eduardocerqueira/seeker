#date: 2022-11-10T17:17:29Z
#url: https://api.github.com/gists/0c2a138fd90825527e6bf63da94984ca
#owner: https://api.github.com/users/leiterenato

from google.cloud import monitoring_v3
import time

client = monitoring_v3.MetricServiceClient()
project_name = f"projects/rl-alphafold-dev"
now = time.time()
seconds = int(now)
nanos = int((now - seconds) * 10 ** 9)
interval = monitoring_v3.TimeInterval(
    {
        "end_time": {"seconds": seconds, "nanos": nanos},
        "start_time": {"seconds": (seconds - 3628800), "nanos": nanos},
    }
)

filter = 'metric.type = "ml.googleapis.com/training/cpu/utilization" AND '
filter += 'resource.labels.job_id = "4411643896526798848" AND '
filter += 'resource.type = "cloudml_job"'

results = client.list_time_series(
    request={
        "name": project_name,
        "filter": filter,
        "interval": interval,
        "view": monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL,
    }
)
for result in results:
    print(result)