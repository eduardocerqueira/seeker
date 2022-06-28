#date: 2022-06-28T16:55:24Z
#url: https://api.github.com/gists/4e1ade3dea14f291988bd6bd229eafe4
#owner: https://api.github.com/users/birinder-lobana

from openscreen import Openscreen
import os
#Obtain your access key and secret from Openscreen Dashboard
ospro = Openscreen(access_key=os.environ.get('OS_ACCESS_KEY'), access_secret=os.environ.get('OS_ACCESS_SECRET'))
#Create a new project on Openscreen Dashboard. Paste the projectId below
projectId = os.environ.get('PROJECT_ID')

assetId = 'xxxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx'

scans = os.asset(assetId).scans().get()
