#date: 2025-07-11T16:55:48Z
#url: https://api.github.com/gists/08d0f1b801ea3aebe4cc47f6bbe2cfae
#owner: https://api.github.com/users/GraceCindie

def merge(left, right):
    res = []
    l_index = 0
    r_index = 0

    while l_index < len(left) and r_index < len(right):
        if left[l_index] < right[r_index]:
            res.append(left[l_index])
            l_index += 1
        else:
            res.append(right[r_index])
            r_index += 1

    res.extend(left[l_index:])
    res.extend(right[r_index:])

    return res


def sort(arr):
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2

    start = arr[mid:]
    end = arr[:mid]

    left = sort(start)
    right = sort(end)

    return merge(left, right)


arr = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]

print(sort(arr)) # Output : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]