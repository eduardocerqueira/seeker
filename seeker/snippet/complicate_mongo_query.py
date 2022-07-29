#date: 2022-07-29T16:58:16Z
#url: https://api.github.com/gists/8a2cd19c05b22ee9a5efc82135c47284
#owner: https://api.github.com/users/re0phimes

# query from list
query = {"$or": [{'data': {'$elemMatch': {'url': {'$exists': False}}}},
                    {'data': {'$elemMatch': {'tag': {'$exists': False}}}}]}

# query not exist
