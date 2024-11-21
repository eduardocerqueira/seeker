#date: 2024-11-21T17:09:41Z
#url: https://api.github.com/gists/e0fd90d8c4ca9d89cb3200558a7c2249
#owner: https://api.github.com/users/sts-developer

import http.client

conn = http.client.HTTPSConnection("testapi.taxbandits.com")
payload = ''
headers = {
  'Authorization': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhdWQiOiJlNDdhN2I3MGMwNTY0NjI2OTU0M2RhNzQwZmNiNTZmNCIsImV4cCI6MTczMTkxMDcxOSwiaWF0IjoxNzMxOTA3MTE5LCJpc3MiOiJodHRwczovL3Ricy1vYXV0aC5zdHNzcHJpbnQuY29tL3YyLyIsInN1YiI6ImJlZWQ0ZTAxYzM2NmQ4MjIiLCJ1c2VydW5pcXVlaWQiOiIifQ.ojZn07OhqPpuVGpcb5wInE-Y5z7IkXqHtpOIRSV8zqo'
}
conn.request("GET", "/V1.7.3/form1099B/Validate?SubmissionId=d259edc3-b59a-4771-926d-1f68269a5473", payload, headers)
res = conn.getresponse()
data = res.read()
print(data.decode("utf-8"))