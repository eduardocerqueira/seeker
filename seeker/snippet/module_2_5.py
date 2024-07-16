#date: 2024-07-16T17:08:15Z
#url: https://api.github.com/gists/0e6f3e88bf14b546482299c18ab83de2
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
