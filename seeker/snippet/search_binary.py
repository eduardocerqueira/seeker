#date: 2021-12-08T17:16:15Z
#url: https://api.github.com/gists/d3cd2762febcf519fc069c30d541bc46
#owner: https://api.github.com/users/fumanchez

from math import floor


def search_binary(collection, element):
    begin, end = 0, len(collection) - 1
    while begin <= end:
        index = floor((begin + end) / 2)
        guess = collection[index]
        if guess == element:
            return index

        if guess < element:
            begin = index + 1
        else:
            end = index - 1

    return None


numbers = [4, 7, 32, 66, 100]
for n in numbers + [3, 5, 100, 111]:
    i = search_binary(numbers, n)
    if i is not None:
        print(f"element '{n}' places at {i} index")
    else:
        print(f"element '{n}' not found in {numbers}")
