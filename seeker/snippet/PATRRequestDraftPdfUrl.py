#date: 2024-11-29T17:11:26Z
#url: https://api.github.com/gists/5121796342011b2674dd41e6e10bd8a1
#owner: https://api.github.com/users/sts-developer

import http.client
import json

conn = http.client.HTTPSConnection("testapi.taxbandits.com")
payload = json.dumps({
  "TaxYear": None,
  "RecordId": "cf0a188b-6661-4b57-b04b-ba9ead52a16e",
  "Business": None,
  "Recipient": None
})
headers = {
  'Authorization': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhdWQiOiJlYzE0NTIxYjMxNGY0N2RhOTc5ODMzYjVlZjkxNDU5ZSIsImV4cCI6MTcyMTIwNjk3NywiaWF0IjoxNzIxMjAzMzc3LCJpc3MiOiJodHRwczovL3Rlc3RvYXV0aC5leHByZXNzYXV0aC5uZXQvdjIvIiwic3ViIjoiYTQyMWE2MWUzOWUyY2U3ZSJ9.YjQP2gkoecj6tqyXJRcJ8LeeAsP0zwkQYk7iD0QHKW8',
  'Content-Type': 'application/json'
}
conn.request("POST", "/v1.7.3/Form1099PATR/RequestDraftPdfUrl", payload, headers)
res = conn.getresponse()
data = res.read()
print(data.decode("utf-8"))