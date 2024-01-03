#date: 2024-01-03T17:04:52Z
#url: https://api.github.com/gists/fad29c23dbb448bfa8f223902e824ba7
#owner: https://api.github.com/users/ag-michael

import os,sys
import requests
import xmltodict
import hmac
import hashlib
import datetime
import base64,time
import subprocess

BHE_TOKEN_ID = "**********"
BHE_TOKEN_KEY = "**********"

PATH = "/path/to/files"


def format_url(uri):
  formatted_uri = uri
  if uri.startswith("/"):
    formatted_uri = formatted_uri[1:]
  return f"http://127.0.0.1:8080/{formatted_uri}"

def _request(method, uri, body=b''):
  global BHE_TOKEN_ID
  global BHE_TOKEN_KEY

  url=format_url(uri)
  digester = None
  
  digester = "**********"

  digester.update(f"{method}{uri}".encode())
  digester = hmac.new(digester.digest(), None, hashlib.sha256)

  datetime_formatted = datetime.datetime.now().astimezone().isoformat("T")
  digester.update(datetime_formatted[:13].encode())
  digester = hmac.new(digester.digest(), None, hashlib.sha256)

  if body is not None:
     digester.update(body)
  # Perform the request with the signed and expected headers
  headers={
         "User-Agent": "Agent User",
         "Authorization": "**********"
         "RequestDate": datetime_formatted,
         "Signature": base64.b64encode(digester.digest()),
         "Content-Type": "application/json",
     }
  return requests.request(
     method=method,
     url=url,
     headers=headers,
     data=body,
    timeout=86400,
  )

def getFile(fname):
  with open(fname,"r",encoding="utf-8") as f:
   return bytes(f.read(),"utf-8")
 
def uploadToBH(fname):
  global PATH
  URL = "http://localhost:8080"
  os.chdir(PATH)
  if fname.lower().endswith(".zip"):
    subprocess.call(['unzip',fname])
  for item in os.listdir(PATH):
    if item.endswith(".json"):
      jsonFile = f"{PATH}{os.sep}{item}"
      print(f"Uploading {jsonFile}")
      sys.stdout.flush()
      uploadResult = _request(method="POST",uri='/api/v2/file-upload/start',body=b'{}')
      print(f"Status code:{uploadResult.status_code}")
      print("Result:\n",uploadResult.text)
      request_id = uploadResult.json()["data"]["id"]
      data = getFile(jsonFile)
      uploadFileResult = _request(method="POST",uri=f'/api/v2/file-upload/{request_id}',body=data)
      print(f"Status code:{uploadFileResult.status_code}")
      print("Result:\n",uploadFileResult.text)
      print(f"\nFile sent: {jsonFile}")
      uploadFileResult = _request(method="POST",uri=f'/api/v2/file-upload/{request_id}/end',body=b'')
      print(f"Status code:{uploadFileResult.status_code}")
      print("Result:\n",uploadFileResult.text)
      print(f"\nFinished uploading: {jsonFile}")
      os.system(f"mv -v {jsonFile} {jsonFile}.done")

if __name__ == "__main__":
  for fname in os.listdir(PATH):
    if fname.lower().endswith(".zip") or fname.lower().endswith(".json"):
      print(f"Found:{PATH}{os.sep}{fname}")
      uploadToBH(f"{PATH}{os.sep}{fname}")
ATH}{os.sep}{fname}")
      uploadToBH(f"{PATH}{os.sep}{fname}")
