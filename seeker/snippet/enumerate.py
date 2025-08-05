#date: 2025-08-05T17:08:28Z
#url: https://api.github.com/gists/b7474568d9474624320b9cbb5d058d1d
#owner: https://api.github.com/users/nikolaydrumev

def enumerate(el, start=0):
    result = []
    index = start
    for item in el:
        result.append((index, item))
        index += 1
    return result

print(enumerate(['n', 'i', 'k', 'i'], start=1))