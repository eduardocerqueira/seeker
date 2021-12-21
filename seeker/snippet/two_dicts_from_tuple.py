#date: 2021-12-21T16:59:52Z
#url: https://api.github.com/gists/3a257374195f1d405f4c10c617de5b97
#owner: https://api.github.com/users/hansalemaos

def two_dicts_from_tuple(tup):
    dict1 ={}
    dict2 = {}
    for t in tup:
        dict1[t[0]] = t[1]
        dict2[t[1]] = t[0]
    return dict1.copy(), dict2.copy()