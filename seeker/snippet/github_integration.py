#date: 2022-10-31T17:29:34Z
#url: https://api.github.com/gists/e9f7c43ddeef1f7599f7225cfa4c83a3
#owner: https://api.github.com/users/Shawnice

from github import GithubIntegration

with open("private-key.pem") as fin:  
    private_key = fin.read()  
  
github_integration = GithubIntegration(os.environ.get("APP_ID"), private_key)