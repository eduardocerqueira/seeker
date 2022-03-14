#date: 2022-03-14T16:49:42Z
#url: https://api.github.com/gists/a98798f1e0650f3783e1c5b91670cccf
#owner: https://api.github.com/users/iblis17

import pygit2

# open existing repository
repo = pygit2.Repository(pygit2.discover_repository('test_repos'))

# branch against tree
branch = repo.lookup_branch('some_branch')
# master would be: repo.head.peel().tree
tree = branch.peel().tree

# use an in-memory index and read current tree
index = pygit2.Index()
index.read_tree(tree)

# add data
blob_id = repo.create_blob('foobar')
entry = pygit2.IndexEntry('full/path/to/file', blob_id, pygit2.GIT_FILEMODE_BLOB)
index.add(entry)

# can now check for conflicts
index.conflicts

# generate diff from current tree to in-memory index
diff = tree.diff_to_index(index)

# diff patch
patch = diff.patch