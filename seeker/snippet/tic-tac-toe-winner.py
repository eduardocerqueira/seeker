#date: 2022-02-23T16:53:08Z
#url: https://api.github.com/gists/e6b80b016c9fdb3340dc37e2847b0fb3
#owner: https://api.github.com/users/Kamforka

def check_rows_for_winner(board):
    for i in range(3):
        row = ""
        for j in range(3):
            row += board[i][j]
        if row == "XXX":
            return "X"
        if row == "OOO":
            return "O"
    return None


def check_columns_for_winner(board):
    for i in range(3):
        col = ""
        for j in range(3):
            col += board[j][i]
        if col == "XXX":
            return "X"
        if col == "OOO":
            return "O"
    return None


def check_first_diagonal_for_winner(board):
    diagonal = ""
    for i in range(3):
        diagonal += board[i][i]
    if diagonal == "XXX":
        return "X"
    if diagonal == "OOO":
        return "O"
    return None


def check_second_diagonal_for_winner(board):
    diagonal = ""
    for i in range(3):
        diagonal += board[i][2 - i]

    if diagonal == "XXX":
        return "X"
    if diagonal == "OOO":
        return "O"
    return None


def get_winner(board):
    winner = check_rows_for_winner(board)
    if not winner:
        winner = check_columns_for_winner(board)
    if not winner:
        winner = check_first_diagonal_for_winner(board)
    if not winner:
        winner = check_second_diagonal_for_winner(board)
    return winner