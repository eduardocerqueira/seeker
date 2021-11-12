#date: 2021-11-12T17:01:30Z
#url: https://api.github.com/gists/df9a823e33ff53ddb1f6b7373c1b1e75
#owner: https://api.github.com/users/ysntrkc

#######################################################################
# 1

flatten_ls = []


def flatten(lis):
    for item in lis:
        if type(item) == list:
            flatten(item)
        else:
            flatten_ls.append(item)

    return flatten_ls


ls = [[1, 'a', ['cat'], 2], [[[3]], 'dog'], 4, 5]
flatten(ls)
print(flatten_ls)


#######################################################################
# 2

def reverse(lis):
    reversed_list = [element[::-1] for element in lis][::-1]

    return reversed_list


ls = [[1, 2], [3, 4], [5, 6, 7]]
ls = reverse(ls)
print(ls)
