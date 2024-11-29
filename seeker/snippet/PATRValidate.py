#date: 2024-11-29T17:08:34Z
#url: https://api.github.com/gists/83fafd0750fb0b6beacc9444a20797b2
#owner: https://api.github.com/users/sts-developer

import http.client

conn = http.client.HTTPSConnection("testapi.taxbandits.com")
payload = ''
headers = {
  'Authorization': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhdWQiOiJlYzE0NTIxYjMxNGY0N2RhOTc5ODMzYjVlZjkxNDU5ZSIsImV4cCI6MTcyMTIwNjk3NywiaWF0IjoxNzIxMjAzMzc3LCJpc3MiOiJodHRwczovL3Rlc3RvYXV0aC5leHByZXNzYXV0aC5uZXQvdjIvIiwic3ViIjoiYTQyMWE2MWUzOWUyY2U3ZSJ9.YjQP2gkoecj6tqyXJRcJ8LeeAsP0zwkQYk7iD0QHKW8'
}
conn.request("GET", "/v1.7.3/Form1099PATR/Validate?SubmissionId=9d71ae45-df5f-49f7-86f8-e88f54132fa1&RecordIds=01132f6d-ef4a-4014-817e-94a5a19bd52b,eb39714e-f653-4c4d-a53c-07f24f9a9dj5", payload, headers)
res = conn.getresponse()
data = res.read()
print(data.decode("utf-8"))