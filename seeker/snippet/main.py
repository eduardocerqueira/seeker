#date: 2025-08-28T17:01:09Z
#url: https://api.github.com/gists/26c42d38edeb3c64d1c0aec3887dcf61
#owner: https://api.github.com/users/matthewelwell

import json

import requests


BASE_URL = "https://api.flagsmith.com/api/v1"

API_KEY = "<api key>"
ENVIRONMENT_KEY = "<environment key>"

EDGE_IDENTITIES_BASE_URL = f"{BASE_URL}/environments/{ENVIRONMENT_KEY}/edge-identities"

session = requests.Session()
session.headers.update(
    {
        "Authorization": f"Api-Key {API_KEY}",
        "Content-Type": "application/json"
    }
)


def update_alias_for_identifier(identifier: str, alias: str) -> None:
    identities_response = session.get(f"{EDGE_IDENTITIES_BASE_URL}?q={identifier}")
    identities_response.raise_for_status()
    results = identities_response.json()["results"]
    assert len(results) == 1
    identity_uuid = results[0]["identity_uuid"]

    update_response = session.put(
        f"{EDGE_IDENTITIES_BASE_URL}/{identity_uuid}/",
        data=json.dumps({"dashboard_alias": alias}),
    )
    update_response.raise_for_status()