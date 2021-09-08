#date: 2021-09-08T17:17:08Z
#url: https://api.github.com/gists/c90e9b31719e36e029ba0587f6ae874c
#owner: https://api.github.com/users/skyprince999

import os
import re
import json

from googleapiclient.discovery import build

from elasticsearch import Elasticsearch
from elasticsearch import helpers

import datetime
import requests
import logging

#set logging 
logging.getLogger('googleapiclient.discovery_cache').setLevel(logging.ERROR)

#read the telegram token from the environment
telegram_token = os.getenv('TELEGRAM_TOKEN')
TELEGRAM_API_HOST = "api.telgram.org"

URL = "https://api.telegram.org/bot{}/".format(telegram_token)

#API token for factcheck API
apiToken = os.getenv('GCP_FACTCHECK_TOKEN')
# Build the factcheck API service as a global var
service = build('factchecktools', 'v1alpha1', developerKey = apiToken)
resource = service.claims()

# Update ES with test queries
hostname = os.getenv('ES_URL')
es = Elasticsearch([hostname], sniff_on_connection_fail= True) #, timeout=500000) << Not certain these parameters add value
