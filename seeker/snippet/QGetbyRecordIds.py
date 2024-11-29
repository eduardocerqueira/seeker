#date: 2024-11-29T17:05:12Z
#url: https://api.github.com/gists/f48b6e78b2723090a65adcd76d84fa18
#owner: https://api.github.com/users/sts-developer

import http.client
import json

conn = http.client.HTTPSConnection("testapi.taxbandits.com")
payload = json.dumps({
  "RecordIds": [
    {
      "RecordId": "7e2b2902-fc16-4dce-bdfd-747617b35c77"
    },
    {
      "RecordId": "7e2b2902-fc16-4dce-bdfd-747617b35c77"
    }
  ]
})
headers = {
  'Authorization': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhdWQiOiJlYzE0NTIxYjMxNGY0N2RhOTc5ODMzYjVlZjkxNDU5ZSIsImV4cCI6MTcyMTIwNjk3NywiaWF0IjoxNzIxMjAzMzc3LCJpc3MiOiJodHRwczovL3Rlc3RvYXV0aC5leHByZXNzYXV0aC5uZXQvdjIvIiwic3ViIjoiYTQyMWE2MWUzOWUyY2U3ZSJ9.YjQP2gkoecj6tqyXJRcJ8LeeAsP0zwkQYk7iD0QHKW8',
  'Content-Type': 'application/json'
}
conn.request("POST", "/v1.7.3/Form1099Q/GetbyRecordIds", payload, headers)
res = conn.getresponse()
data = res.read()
print(data.decode("utf-8"))