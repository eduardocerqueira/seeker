#date: 2025-06-02T16:50:53Z
#url: https://api.github.com/gists/ba5e57ba5d7e02e8c46c73d2d015a987
#owner: https://api.github.com/users/bolablg

import gspread
from google.oauth2 import service_account
from googleapiclient.discovery import build

scopes = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive'
]

sa_credentials = service_account.Credentials.from_service_account_file(chemin/vers/ta_cle.json')

creds = sa_credentials.with_scopes(scopes)
gw_client = gspread.authorize(credentials = creds)