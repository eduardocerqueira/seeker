#date: 2023-10-23T16:41:56Z
#url: https://api.github.com/gists/cb66abec0bd24e940e7abffde8a3645f
#owner: https://api.github.com/users/saurabh-hirani

import sys
import json
import logging
import os
import random
import time
from os.path import splitext, basename

import opentelemetry.metrics as metrics
import psutil
from opentelemetry.metrics import Observation, CallbackOptions
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.exporter.prometheus_remote_write import (
    PrometheusRemoteWriteMetricsExporter,
)
from opentelemetry.sdk.metrics.view import View
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.metrics._internal.aggregation import ExponentialBucketHistogramAggregation

view = View(instrument_name="histogram", aggregation=ExponentialBucketHistogramAggregation())

cpu_gauge = None
ram_gauge = None

# Callback to gather cpu usage
def get_cpu_usage_callback(_: CallbackOptions):
    for (number, percent) in enumerate(psutil.cpu_percent(percpu=True)):
        attributes = {"cpu_number": str(number)}
        yield Observation(percent, attributes)


# Callback to gather RAM memory usage
def get_ram_usage_callback(_: CallbackOptions):
    ram_percent = psutil.virtual_memory().percent
    yield Observation(ram_percent)


if __name__ == '__main__':
    script_name = splitext(basename(__file__))[0]

    loglevel = os.environ.get("LOGLEVEL", "INFO").upper()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(script_name)

    exporter = PrometheusRemoteWriteMetricsExporter(
        endpoint=sys.argv[1],
        basic_auth={
            "username": sys.argv[2],
            "password": "**********"
        },
        headers={},
    )
    metric_prefix = sys.argv[4]

    reader = PeriodicExportingMetricReader(exporter, 30 * 1000)
    provider = MeterProvider(metric_readers=[reader], views=[view])
    metrics.set_meter_provider(provider)
    meter = metrics.get_meter(__name__)

    logger.info("creating instruments to record metrics data")
    requests_counter = meter.create_counter(
        name=metric_prefix + "_requests", description="number of requests", unit="1"
    )

    requests_size = meter.create_histogram(
        name=metric_prefix + "_request_size", description="size of requests", unit="byte"
    )

    cpu_gauge = meter.create_observable_gauge(
        callbacks=[get_cpu_usage_callback], name=metric_prefix + "_cpu_percent", description="per-cpu usage", unit="1"
    )

    ram_gauge = meter.create_observable_gauge(
        callbacks=[get_ram_usage_callback],
        name=metric_prefix + "_ram_percent",
        description="RAM memory usage",
        unit="1",
    )

    labels = json.loads(sys.argv[5])

    logger.info("starting")
    try:
        while True:
            logger.info("updating")
            requests_counter.add(random.randint(0, 30), labels)
            requests_size.record(random.randint(10, 100), labels)
            time.sleep(10)
    except KeyboardInterrupt:
        logger.info("shutting down")