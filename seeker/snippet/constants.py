#date: 2021-11-15T17:07:55Z
#url: https://api.github.com/gists/fc3f8f3a64cbb04b9dc6486195ceb95d
#owner: https://api.github.com/users/Steboss89

import os
import tarfile
import datetime
import logging
from mlflow.tracking.client import MlflowClient
from mlflow.deployments import BaseDeploymentClient
from google.cloud import storage
from googleapiclient import discovery
import mlflow_aiplatform

# GLOBAL VARIABLES
AI_PLATFORM = discovery.build('ml','v1')
GS_BUCKET = storage.Client()
TODAY = datetime.datetime.today().strftime('%Y-%m-%dT%H:%M:%SZ')
# retrieve pacakge files, so predictor and setup can be used as template
MLFLOW_AIPLATFORM_PATH = mlflow_aiplatform.__path__[0]
# define logger
logger = logging.getLogger(__name__)

# TODO use PATH to safely create paths

def run_local(name, model_uri, flavor=None, config=None):
    logger.info("Use `mlflow models serve` to run this model locally")


def target_help():
    return "Deploy MLflow models to aiplatform"