#date: 2025-08-05T17:08:28Z
#url: https://api.github.com/gists/b7474568d9474624320b9cbb5d058d1d
#owner: https://api.github.com/users/nikolaydrumev

def any(iterable): #поне едно да е тру
    for item in iterable:
        if item:
            return True
    return False

print(any([0, '', None, 5]))
print(any([0, '', None]))
