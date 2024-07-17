#date: 2024-07-17T16:49:10Z
#url: https://api.github.com/gists/5a85aeba6500ba21709aaa1b457d715a
#owner: https://api.github.com/users/MaXVoLD

def get_matrix(n, m, value):
    matrix = []
    for i in range(1, n + 1):
        list_ = []
        matrix.append(list_)
        for j in range(1, m + 1):
            list_.append(value)
    print(matrix)


get_matrix(2, 4, 10)
