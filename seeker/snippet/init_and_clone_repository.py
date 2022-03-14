#date: 2022-03-14T16:49:42Z
#url: https://api.github.com/gists/a98798f1e0650f3783e1c5b91670cccf
#owner: https://api.github.com/users/iblis17

import pygit2

# bare repository
repo = pygit2.init_repository('test_repos', True)

# normal repository
repo = pygit2.init_repository('test_repos')

# clone repository
repo = pygit2.clone_repository('path_or_remote_host', 'new_cloned_repo', bare=True)

