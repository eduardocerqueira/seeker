#date: 2025-11-04T17:09:03Z
#url: https://api.github.com/gists/1509b96c8877dae5129aa605f3dd193f
#owner: https://api.github.com/users/santosh-gouda

#!/usr/bin/env python
# https://gist.github.com/eldondevcg/fffff4b7909351b19a53
import datetime as dt
import json
import os
import time

import boto3
import click


def get_last_jr(job_name):
    client = boto3.client("glue")

    response = client.get_job_runs(
        JobName=job_name,
    )

    if response.get("JobRuns"):
        job_runs = response.get("JobRuns")
        if len(job_runs) > 0:
            return response["JobRuns"][0]["Id"]

    return None


def get_job_streams(job_name, job_run):

    glue_client = boto3.client("glue")

    check_job = glue_client.get_job_run(JobName=job_name, RunId=job_run)

    if not check_job["JobRun"]["JobRunState"] in [
        "STOPPING",
        "STOPPED",
        "SUCCEEDED",
        "FAILED",
        "TIMEOUT",
        "ERROR",
    ]:
        return True
    else:
        logs_client = boto3.client("logs")

        group_names = [
            "/aws-glue/jobs/error",
            "/aws-glue/jobs/logs-v2",
            "/aws-glue/jobs/output",
        ]

        for group_name in group_names:
            try:
                logs_batch = logs_client.get_log_events(
                    logGroupName=group_name, logStreamName=job_run
                )

                print(json.dumps(logs_batch, indent=2))
            except BaseException:
                print(f"{ group_name } not found!")
        return False


def get_last_streams():
    group_names = [
        "/aws-glue/jobs/error",
        "/aws-glue/jobs/logs-v2",
        "/aws-glue/jobs/output",
    ]
    client = boto3.client("logs")

    for group_name in group_names:
        stream_response = client.describe_log_streams(
            logGroupName=group_name, orderBy="LastEventTime", limit=1
        )

        latestlogStreamName = stream_response["logStreams"][0]["logStreamName"]

        stream_batch = client.get_log_events(
            logGroupName=group_name,
            logStreamName=latestlogStreamName,
            startTime=int((dt.datetime.today() - dt.timedelta(minutes=15)).timestamp()),
            endTime=int(dt.datetime.now().timestamp()),
        )

        all_streams = []
        stream_batch = client.describe_log_streams(
            logGroupName=group_name,
            orderBy="LastEventTime",
        )
        all_streams += stream_batch["logStreams"]

        stream_names = [stream["logStreamName"] for stream in all_streams]

        for stream in stream_names:
            logs_batch = client.get_log_events(
                logGroupName=group_name,
                logStreamName=stream,
                startTime=int(
                    (dt.datetime.today() - dt.timedelta(minutes=5)).timestamp()
                ),
                endTime=int(dt.datetime.now().timestamp()),
            )
            for event in logs_batch["events"]:
                event.update({"group": group_name, "stream": stream})
                print(json.dumps(event, indent=2))


def inject_requirements(filename="/requirements/spark-base.txt"):

    reqs = []
    with open(os.getcwd() + filename, "r") as f:
        content = f.readlines()

    for line in content:
        if not line.__contains__("#"):
            reqs.append(line.replace("\n", ""))

    return ",".join(reqs)


def submit_run_job(name):
    client = boto3.client("glue")
    response = client.start_job_run(
        JobName=name,
        Arguments={
            "--REGION": "eu-central-1",
            "--SECRET_NAME": "**********"
            "--TempDir": "s3://datalake.caronsale.cloud/staging",
            "--job-bookmark-option": "job-bookmark-disable",
            "--additional-python-modules": inject_requirements(),
        },
    )

    if response["ResponseMetadata"]["HTTPStatusCode"] == 200:
        print("Request was submitted successfully")
    else:
        print(response)


@click.group()
def cli():
    """A Simple utility to retrive Glue logs from Cloudwatch"""
    pass


@cli.command()
@click.option("-n", "--name", prompt="Job name")
def get_logs(name):
    """Retrives the logs of the last run for a specific job"""
    job_id = get_last_jr(name)
    get_job_streams(name, job_id)


@cli.command()
@click.option("-n", "--name", prompt="Job name")
def monitor_logs(name):
    """Monitors the logs of the last run for a specific job, 15 seconds hit"""
    job_id = get_last_jr(name)
    while get_job_streams(name, job_id):
        get_job_streams(name, job_id)
        time.sleep(15)


@cli.command()
def get_last():
    """Retrives Glue logs that occured in the last 15 minutes"""
    get_last_streams()


@cli.command()
@click.option("-n", "--name", prompt="Job name")
def run_job(name):
    """Run a job"""
    submit_run_job(name)


if __name__ == "__main__":
    cli()


