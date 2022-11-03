#date: 2022-11-03T17:10:54Z
#url: https://api.github.com/gists/14859f9fbb73a234d71a6cef1cea73ea
#owner: https://api.github.com/users/shagreel

import json
import requests

def requestJourneyData():
    userToken = "**********"
    imsOrg = '<IMS_ORG_HERE>'
    sandboxName = '<SANDBOX_NAME_HERE>'
    apiKey = '<API_KEY_HERE>'

    url = 'https://journey-private.adobe.io/authoring/journeyVersions'
    headers = { 'Authorization': "**********"
                'Content-Type': 'application/json', 
                'x-api-key': apiKey, 
                'x-gw-ims-org-id': imsOrg, 
                'x-sandbox-name': sandboxName}

    request = requests.get(url, headers=headers)

    data = request.json()
    print(json.dumps(data))
    return data['results']
    

def printJourneyData(data, journeyVersionId=None):

    for journey in data:
        if journeyVersionId is not None and journey['_id'] != journeyVersionId:
            continue
        
        output = []
        for step in journey['steps']:
            if 'nodeType' in step and step['nodeType'] in ['message', 'action']:
                if 'uid' in step['action']:
                    msg = {'_id': step['action']['uid'], '_ringcentral': {'nodeName': step['nodeName'], 'journeyName': journey['name']} }
                    output.append(msg)

        for row in output:
            print(json.dumps(row))

data = requestJourneyData()

# Get a sinlge journey
journeyVersionId = '<Journey Version UUID>'
printJourneyData(data, journeyVersionId)

# Get all journeys
#printJourneyData(data)tJourneyData(data)