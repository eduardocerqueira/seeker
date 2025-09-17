#date: 2025-09-17T16:53:37Z
#url: https://api.github.com/gists/d63c2b476f7ab301af6a1f6bcc7f9efc
#owner: https://api.github.com/users/rodion-gudz

GITHUB_USERNAME=rodion-gudz; curl https://api.github.com/users/$GITHUB_USERNAME/repos?per_page=1000 | jq .[]."clone_url" | xargs -n1 git clone