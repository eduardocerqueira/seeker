#date: 2022-03-14T16:49:42Z
#url: https://api.github.com/gists/a98798f1e0650f3783e1c5b91670cccf
#owner: https://api.github.com/users/iblis17

import pygit2

# open existing repository
repo = pygit2.Repository(pygit2.discover_repository('test_repos'))

# read index
repo.index.read()

for commit in repo.walk(repo.head.target, pygit2.GIT_SORT_TOPOLOGICAL):
    parent_oid = None

    if len(commit.parents) > 0:
        parent_oid = commit.parents[0].oid

    print("{} commit: {} - parents: {}".format(commit.oid, commit.message.strip(), parent_oid))