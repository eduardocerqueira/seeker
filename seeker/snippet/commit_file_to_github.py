#date: 2023-12-11T16:56:35Z
#url: https://api.github.com/gists/925da6ef98001a815f68262a0f5b6cbc
#owner: https://api.github.com/users/sreehari1997

import base64
import requests


class GitHubRepo:
    def __init__(self):
        self.base_url = "https://api.github.com/repos"
        self.repo = "GITHUB_REPOSITORY_NAME"
        self.owner = "GITHUB_USERNAME"
        self.token = "**********"
        self.branch = "BRANCH_NAME"
        self.email = "EMAIL"

    def get_headers(self):
        headers = {
            'Authorization': "**********"
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 6.1; Win64; '
                'x64; rv:47.0) Gecko/20100101 Firefox/47.0'
            ),
            'X-GitHub-Api-Version': '2022-11-28',
            'Accept': 'application/vnd.github+json'
        }
        return headers

    def put(self, url, data):
        try:
            response = requests.put(
                url,
                headers=self.get_headers(),
                json=data
            )
            return response.json()
        except requests.exceptions.RequestException:
            return None

    def create_file(self, content_in_bytes, path, commit_message):
        encoded = str(base64.b64encode(content_in_bytes).decode("utf-8"))
        data = {
            "message": commit_message,
            "committer": {
                "name": self.owner,
                "email": self.email
            },
            "content": encoded
        }
        url = "{}/{}/{}/contents/{}".format(
            self.base_url,
            self.owner,
            self.repo,
            path
        )
        response = self.put(url, data)
        return responseata)
        return response