#date: 2023-03-20T16:51:02Z
#url: https://api.github.com/gists/5571b601c114d4a24c0e8d2c3e9e7e5d
#owner: https://api.github.com/users/Shelob9

import api

gh = api.Github('imaginarymachines', 'small')
indexer = api.Indexer()
indexer.index_documents(gh.documents,'outputs/gh/small.json')
