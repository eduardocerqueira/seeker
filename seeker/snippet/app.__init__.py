#date: 2022-06-17T17:02:31Z
#url: https://api.github.com/gists/471bc58e08abcddfb2d6229a40b5aa9a
#owner: https://api.github.com/users/katrbhach

import json
from flask import Flask
from prometheus_flask_exporter import PrometheusMetrics
from logging.config import dictConfig

dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "[%(asctime)s] [%(levelname)s] %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S %z"
            }
        },
        "handlers": {
            "wsgi": {
                "class": "logging.StreamHandler",
                "stream": "ext://flask.logging.wsgi_errors_stream",
                "formatter": "default",
            }
        },
        "root": {"level": "INFO", "handlers": ["wsgi"]},
    }
)

app = Flask(__name__)
metrics = PrometheusMetrics(app)
mrc_config = json.load(open("/etc/conf/mrc-config.json"))

# static information as metric
metrics.info('app_info', 'Application info', version='1.0.3')

# Load the views
from app import topic_views, health_view
