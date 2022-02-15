#date: 2022-02-15T17:03:03Z
#url: https://api.github.com/gists/a8cde23221f66f1864387fd7506683ca
#owner: https://api.github.com/users/evangelos-zafeiratos

# Store in a dictionary the legitimate values for each individual cell
def cacheValidValues(board):
    cache = dict()
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:
                cache[(i,j)] = allowedValues(board,i,j)
    return cache