#date: 2023-07-27T17:00:13Z
#url: https://api.github.com/gists/22b0a6f10fbc220b5ee30e07a59398b1
#owner: https://api.github.com/users/cinder-star

result = []

string = input()

change_arr = [i for i in range(len(string))]


def string_builder():
    result = ""
    for i, val in enumerate(change_arr):
        if val == 0:
            result += string[i].lower()
        else:
            result += string[i].upper()
    return result


def recursion(n):
    if n < 0:
        result.append(string_builder())
        return
    if string[n].isdecimal():
        recursion(n - 1)
    else:
        change_arr[n] = 0
        recursion(n - 1)
        change_arr[n] = 1
        recursion(n - 1)


recursion(len(string) - 1)

print(result)
