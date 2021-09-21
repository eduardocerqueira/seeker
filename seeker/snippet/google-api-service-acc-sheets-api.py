#date: 2021-09-21T16:59:39Z
#url: https://api.github.com/gists/9e6fed0f7c95a27a14817b5778f0c176
#owner: https://api.github.com/users/Denis070

import os

import httplib2
from googleapiclient.discovery import build
from oauth2client.service_account import ServiceAccountCredentials

"""
Source for https://youtu.be/NgMoz50no6I
"""

def get_service_sacc():
    """
    Create a project in Google Cloud.
    Then create a service account.
    Cope service account json to local file.
    You can download the jcion file only when you create a service account!
    And copy the service account email. 	some-service-acc@xxxx.com
    And give the service account access to the spreadsheet.
    :return:
    """
    creds_json = os.path.dirname(__file__) + "/creds/some-acc-key.json"

    scopes = [
        'https://www.googleapis.com/auth/spreadsheets'
    ]

    creds_service = ServiceAccountCredentials.\
        from_json_keyfile_name(creds_json, scopes).\
        authorize(httplib2.Http())

    return build('sheets', 'v4', http=creds_service)

# appending data to spreadsheet
get_service_sacc().spreadsheets() \
    .values() \
    .append(
    spreadsheetId="xxxx",
    range="Sheet1!A:Z",
    valueInputOption="RAW",
    body={'values' : [['Hola, AzzraelCode YouTube subs!!!']] }
).execute()

