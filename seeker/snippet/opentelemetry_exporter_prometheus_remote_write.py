#date: 2021-12-13T16:56:33Z
#url: https://api.github.com/gists/d5857fe8241431a5dda9b0dbb767959d
#owner: https://api.github.com/users/aengusrooneygrafana

#!/usr/bin/python3

from flask import Flask
from flask import render_template, Response, request

import json
import time

from opentelemetry import metrics
from opentelemetry.exporter.prometheus_remote_write import (
    PrometheusRemoteWriteMetricsExporter,
)
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export.aggregate import (
    HistogramAggregator,
    LastValueAggregator,
    MinMaxSumCountAggregator,
    SumAggregator,
)
from opentelemetry.sdk.metrics.view import View, ViewConfig

app = Flask(__name__)

def send_metrics(deviceid, name, value):
    metrics.set_meter_provider(MeterProvider())
    meter = metrics.get_meter(__name__)
    exporter = PrometheusRemoteWriteMetricsExporter(
      endpoint="https://prometheus-prod-10-prod-us-central-0.grafana.net/api/prom/push",
      basic_auth={"username": "<username>", "password": "<token>",},
    )
    testing_labels = {"deviceid":deviceid}

    metrics.get_meter_provider().start_pipeline(meter, exporter, 1)

    requests_active = meter.create_updowncounter(
      name=name,
      description=name,
      unit="1",
      value_type=float,
    )
    requests_active.add(float(value), testing_labels)

# Example API
@app.route("/ds/v1/sendmetrics", methods = ['POST'])
def sendmetrics():

    data = json.loads(request.data)

    #token = data.get("token", None)
    deviceid = data.get("deviceid", None)
    temp = data.get("temp",None)
    co2 = data.get("co2",None)

    if temp is None or co2 is None or deviceid is None:
        return json.dumps({'msg':False}), 200, {'ContentType':'application/json'}

    send_metrics(deviceid, "temp", temp)
    send_metrics(deviceid, "co2", co2)

    print("deviceid: " + str(deviceid))
    print("temp: " + str(temp))
    print("co2:  " + str(co2))

    return json.dumps({'success':True}), 200, {'ContentType':'application/json'}

if __name__ == "__main__":
    app.run(host='0.0.0.0', port='8450', debug=True)
    app.run()