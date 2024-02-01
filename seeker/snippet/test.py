#date: 2024-02-01T17:00:17Z
#url: https://api.github.com/gists/4010ebfabb1499407246fe3a279b8252
#owner: https://api.github.com/users/hermzz

import hashlib
import datetime
import sys
import jwt
import requests
from typing import Any, Mapping, Optional, Sequence

# Grab from jira integration config, I think it's usually `<sentry_host>.jira`
JIRA_KEY = "sentry.example.com.jira"

# Run `pg_dump -U postgres -t sentry_integration` and then grab `shared_secret` from metadata field for the jira integration
SHARED_SECRET = "**********"

jira_host = 'https://example.atlassian.net'
jira_ticket = 'ABC-1234'

path = f"/rest/api/2/issue/{jira_ticket}"
method = 'GET'
url_params = {}

# Pulled from https://github.com/getsentry/sentry/blob/master/src/sentry/integrations/utils/atlassian_connect.py#L23
def get_query_hash(
    uri: str, method: str, query_params: Mapping[str, str | Sequence[str]] | None = None
) -> str:
    # see
    # https://developer.atlassian.com/static/connect/docs/latest/concepts/understanding-jwt.html#qsh
    uri = uri.rstrip("/")
    method = method.upper()
    if query_params is None:
        query_params = {}

    sorted_query = []
    for k, v in sorted(query_params.items()):
        # don't include jwt query param
        if k != "jwt":
            if isinstance(v, str):
                param_val = percent_encode(v)
            else:
                param_val = ",".join(percent_encode(val) for val in v)
            sorted_query.append(f"{percent_encode(k)}={param_val}")

    query_string = "{}&{}&{}".format(method, uri, "&".join(sorted_query))
    print(f"Query string {query_string}")
    return hashlib.sha256(query_string.encode("utf8")).hexdigest()

# https://github.com/getsentry/sentry/blob/master/src/sentry/integrations/jira/client.py#L77
jwt_payload = {
    "iss": JIRA_KEY,
    "iat": datetime.datetime.utcnow(),
    "exp": datetime.datetime.utcnow() + datetime.timedelta(seconds=5 * 60),
    "qsh": get_query_hash(path, method.upper(), url_params),
}

print(jwt_payload)

encoded_jwt = "**********"="HS256", headers={})

print(encoded_jwt)

print("\n\nRequesting JIRA\n")
response = requests.get(
    f"{jira_host}/rest/api/2/issue/{jira_ticket}?jwt={encoded_jwt}",
    headers={'Content-Type': "application/json"}
)

print(response.status_code)
print(response.text)code)
print(response.text)