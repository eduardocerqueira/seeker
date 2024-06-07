#date: 2024-06-07T17:02:44Z
#url: https://api.github.com/gists/a09205d535d35449381f935454d6707a
#owner: https://api.github.com/users/Leshka60

def get_matrix(n, m, value):
    matrix = []
    for i in range(n):
        matrix.append([])
        for j in range(m):
            matrix[i].append(value)
    return matrix

result1 = get_matrix(2 ,2, 10)
result2 = get_matrix(3, 5, 42)
result3 = get_matrix(4, 2, 13)

print(result1)
print(result2)
print(result3)

