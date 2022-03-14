#date: 2022-03-14T16:49:42Z
#url: https://api.github.com/gists/a98798f1e0650f3783e1c5b91670cccf
#owner: https://api.github.com/users/iblis17

import pygit2

# open existing repository
repo = pygit2.Repository(pygit2.discover_repository('test_repos'))