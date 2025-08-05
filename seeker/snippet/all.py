#date: 2025-08-05T17:08:28Z
#url: https://api.github.com/gists/b7474568d9474624320b9cbb5d058d1d
#owner: https://api.github.com/users/nikolaydrumev

def all(items): #всички да са тру
    for item in items:
        if not item:
            return False
    return True

print(all([1, True, 'ff']))
print(all([10, 0, 'nuae']))