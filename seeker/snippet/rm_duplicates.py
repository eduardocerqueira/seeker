#date: 2022-11-07T17:22:08Z
#url: https://api.github.com/gists/3c1217aa3f4eab0221372fa85dcb8a82
#owner: https://api.github.com/users/klykovdg

def remove_duplicate(arr, s=None, start=0):
    s = s or set()
    for i in range(start, len(arr)):
        keys = ''.join(arr[i].keys())
        if keys not in s:
            s.add(keys)
        else:
            del arr[i]
            remove_duplicate(arr, s, i)
            break


if __name__ == '__main__':
    arr = [{"key1": "value1"}, {"k1": "v1", "k2": "v2", "k3": "v3"}, {}, {},
           {"key1": "value1"}, {"key1": "value1"}, {"key2": "value2"}]
    remove_duplicate(arr)
    print(arr)
    