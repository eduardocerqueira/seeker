#date: 2025-08-05T17:08:28Z
#url: https://api.github.com/gists/b7474568d9474624320b9cbb5d058d1d
#owner: https://api.github.com/users/nikolaydrumev

def zip(*args):
    result = []
    min_length = len(args[0])

    for item in args:
        if len(item) < min_length:
            min_length = len(item)

    for i in range(min_length):
        group = []
        for item in args:
            group.append(item[i])
        result.append(group)

    return result


a = [1, 2, 3, 4]
b = ['n', 'i', 'k', 'i']
print(zip(a, b))
