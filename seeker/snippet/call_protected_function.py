#date: 2023-02-01T16:45:33Z
#url: https://api.github.com/gists/7af83d2f43a51f95d2ac6d35bd611dd2
#owner: https://api.github.com/users/BerangerN

import urllib

import google.auth.transport.requests
import google.oauth2.id_token


def call(endpoint, audience):
    
    req = urllib.request.Request(endpoint)

    auth_req = google.auth.transport.requests.Request()
    id_token = "**********"

    req.add_header("Authorization", f"Bearer {id_token}")
    response = urllib.request.urlopen(req)

    print("RESPONSE ", response)
    return response.read()

def main(request):
    function_1_url "https://project-region-projectid.cloudfunctions.net/function_1"
    return call(function_1_url, function_1_url)  return call(function_1_url, function_1_url)