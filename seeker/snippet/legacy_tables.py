#date: 2023-08-30T17:02:22Z
#url: https://api.github.com/gists/b13bae31f795ca115dfc020b8b90e879
#owner: https://api.github.com/users/felipereyel

import json, requests
from typing import Dict

BASE_URL = "https://tables.abstra.cloud"


def get_headers(api_key: str) -> Dict:
    return {"Api-Authorization": f"Bearer {api_key}"}


class Tables:
    def __init__(self, api_key=None):
        self.api_key = api_key

    def statement(self, id):
        return Statements(id, self.api_key)

    def run_statement(self, id, params=None):
        statement = self.statement(id)
        return statement.run(params)


class Statements:
    def __init__(self, id, api_key=None):
        self.id = id
        self.api_key = api_key

    def run(self, params=None):
        if params is None:
            params = {}
        res = requests.post(
            f"{BASE_URL}/execute/{self.id}",
            json=params,
            headers=get_headers(self.api_key),
        )
        if res.ok:
            return res.json()
        else:
            raise Exception(res.text)