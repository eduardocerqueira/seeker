#date: 2022-01-26T17:07:00Z
#url: https://api.github.com/gists/954b7ba8499c5b2aa9b5862043a97563
#owner: https://api.github.com/users/malfet

#!/usr/bin/env  python3

# Author of https://github.com/pytorch/pytorch/pull/71735 is me (malfet)
# And if executed with my PAT it will return my association as "MEMBER"
# But if executed by somebody outside of org, it will say that association is "CONTRIBUTOR"
# Instead of getting PAT, one can simply execute the query at https://docs.github.com/en/graphql/overview/explorer
GH_GET_ASSOCIATION = """
query {
  repository(owner: "pytorch", name: "pytorch") {
    pullRequest(number: 71735) {
      author {
        login
      }
      authorAssociation
    }
  }
}
"""


def gh_graphql(query):
    import os
    import json
    from urllib.request import urlopen, Request
    token = os.getenv("GITHUB_TOKEN")
    headers = {"Accept": "application/vnd.github.v3+json", "Authorization": f"token {token}"}
    data = json.dumps({"query": query}).encode()
    with urlopen(Request("https://api.github.com/graphql", data=data, headers=headers)) as conn:
        return json.load(conn)


if __name__ == "__main__":
    print(gh_graphql(GH_GET_ASSOCIATION))
