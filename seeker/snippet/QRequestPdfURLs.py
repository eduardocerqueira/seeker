#date: 2024-11-29T16:58:40Z
#url: https://api.github.com/gists/7651d7c653ee1bac2c4c160809fb6591
#owner: https://api.github.com/users/sts-developer

import http.client
import json

conn = http.client.HTTPSConnection("testapi.taxbandits.com")
payload = json.dumps({
  "SubmissionId": "81b77217-fb5a-4315-b76e-bbb805676a38",
  "RecordIds": [
    {
      "RecordId": "ea13cc94-d25f-4c2f-aef2-a97b3eb59cf1"
    }
  ],
  "Customization": {
    "TINMaskType": "Both"
  }
})
headers = {
  'Authorization': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhdWQiOiJlYzE0NTIxYjMxNGY0N2RhOTc5ODMzYjVlZjkxNDU5ZSIsImV4cCI6MTcyMTIwNjk3NywiaWF0IjoxNzIxMjAzMzc3LCJpc3MiOiJodHRwczovL3Rlc3RvYXV0aC5leHByZXNzYXV0aC5uZXQvdjIvIiwic3ViIjoiYTQyMWE2MWUzOWUyY2U3ZSJ9.YjQP2gkoecj6tqyXJRcJ8LeeAsP0zwkQYk7iD0QHKW8',
  'Content-Type': 'application/json'
}
conn.request("POST", "/v1.7.3/Form1099CQRequestPdfURLs", payload, headers)
res = conn.getresponse()
data = res.read()
print(data.decode("utf-8"))