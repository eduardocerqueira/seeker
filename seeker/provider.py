import requests
from os import getenv
from os.path import join
from seeker.util import get_config
from github import Github


class Gists:
    _config = "gists"
    filter = get_config(_config, "language")
    url = get_config(_config, "url")
    user = get_config(_config, "user")

    def __init__(self):
        assert getenv("GITHUB_TOKEN"), "Missing github TOKEN env variable"

    def get(self):
        session = requests.Session()
        session.auth = (self.user, getenv("TOKEN"))
        headers = {"Accept": "application/vnd.github.v3+json"}
        session.headers.update(headers)
        resp = session.get(self.url, headers=headers).json()

        for gist in resp:
            comment = "#"
            for k, v in gist['files'].items():
                if v['language'] in self.filter:
                    if str(v['language']).lower() in ["go", "java"]:
                        comment = "//"
                    file_content = session.get(v['raw_url'])
                    with open(join("./snippet", v['filename']), 'w') as snippet:
                        data_header = f"{comment}date: {gist['created_at']}\n" \
                                      f"{comment}url: {gist['url']}\n" \
                                      f"{comment}owner: {gist['owner']['url']}"
                        data_body = f"{data_header}\n\n{file_content.text}"
                        snippet.write(data_body)


class GitHub:
    _config = "github"
    gh = Github(login_or_token=getenv("TOKEN"))

    def get_repo(self):
        repos = self.gh.search_repositories(query='language:python', sort='stars', order='desc')
        for repo in repos:
            if "public-apis" in repo.full_name:
                continue
            print(f"{repo.full_name} {repo.stargazers_count} {repo.forks_count} {repo.language}")

    # repos = gh.search_repositories(query='filename:performance', sort='stars', order='desc')
    # for repo in repos:
    #     if "public-apis" in repo.full_name:
    #         continue
    #     print(
    #         f"repo: {repo.full_name} "
    #         f"stars: {repo.stargazers_count} "
    #         f"forks: {repo.forks_count} "
    #         f"language: {repo.language}")
