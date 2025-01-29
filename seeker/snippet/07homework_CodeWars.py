#date: 2025-01-29T17:09:24Z
#url: https://api.github.com/gists/3e6044f1d2bd1aae206b358ed233d9fc
#owner: https://api.github.com/users/voyanimg168

[TASK 1]
def invert(lst):
    if lst:
        return [-x for x in lst]
    else:
        return []
    
[TASK 2]
def double_every_other(lst):
    for i in range(1, len(lst), 2):
        lst[i] *= 2
    return lst