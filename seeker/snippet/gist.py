#date: 2025-08-01T17:17:17Z
#url: https://api.github.com/gists/c0cb3ea5ce0b71b3348a65af29ec1544
#owner: https://api.github.com/users/barzugini

n, m = map(int, input().split())

matrix = [[1 + i + j * n for j in range(m)] for i in range(n)]
    
for row in matrix:
    print(*(str(x).ljust(3) for x in row), sep='')