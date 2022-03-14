#date: 2022-03-14T16:49:42Z
#url: https://api.github.com/gists/a98798f1e0650f3783e1c5b91670cccf
#owner: https://api.github.com/users/iblis17

import pygit2

# open existing repository
repo = pygit2.Repository(pygit2.discover_repository('test_repos'))

# check if repos is newly created
if repo.head_is_unborn:
    tb = repo.TreeBuilder()
    parent = []
else:
    tb = repo.TreeBuilder(repo.head.peel().tree.id)
    parent = [repo.head.target]

# add file from anywhere on disk
blob_id = repo.create_blob_fromdisk("/path/to/file")

# insert into tree, and get current tree
# CAVEAT: when adding to a nested path, will need to walk the whole tree inserting new trees
# to get back their oid, and from that oid continue all operations. all operations are "nested", so,
# you will need to grab the last tree from TreeBuilder.write() to make the next change.
# https://gist.github.com/uniphil/9570964 illustrates how to do it, from method auto_insert()
# deleting from a nested path would also require walking the entire tree, like in auto_insert(), but, replacing
# the treebuilder.insert() from the base case to a treebuilder.remove().
tb.insert('file', blob_id, pygit2.GIT_FILEMODE_BLOB)
tree = tb.write()

# add blob from any data
new_blob_id = repo.create_blob('foobar');

# needs to get another treebuilder based on last operation
tb = repo.TreeBuilder(tree)
tb.insert('other_file', new_blob_id, pygit2.GIT_FILEMODE_BLOB)
tree = tb.write()

# delete a file
tb = repo.TreeBuilder(tree)
tb.remove('old_file')
tree = tb.write()

# write to index
repo.index.write()

# author of commit
author = pygit2.Signature('Author', 'author@email.com')

# commit changes
oid = repo.create_commit('refs/heads/master', author, author, 'commit message', tree, parent)