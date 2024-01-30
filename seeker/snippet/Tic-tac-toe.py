#date: 2024-01-30T17:09:00Z
#url: https://api.github.com/gists/5a452adefefa6f3dfc433bd11e54eb2b
#owner: https://api.github.com/users/Vasile70Tanasa

from random import randrange

initial = [['1', '2', '3'], ['4', 'X', '6'], ['7', '8', '9']]


def display_board(board):
    a = ('-'*7).join(['+']*4)
    b = (' '*7).join(['|']*4)
    c0 = '|   '+board[0][0] + \
         '   |   ' + board[0][1] + \
         '   |   ' + board[0][2]+'   |'
    c1 = '|   ' + board[1][0] + \
         '   |   ' + board[1][1] + \
         '   |   ' + board[1][2] + '   |'
    c2 = '|   ' + board[2][0] + \
         '   |   ' + board[2][1] + \
         '   |   ' + board[2][2] + '   |'

    print('\n'.join([a, b, c0, b, a, b, c1, b, a, b, c2, b, a]))


def enter_move(board):
    while True:
        user_move = input('Your move:')
        if user_move in make_list_of_free_fields(board):
            break
        else:
            print('Invalid position!')
    for i in range(3):
        for j in range(3):
            if board[i][j] == user_move:
                board[i][j] = '0'

    display_board(board)

    if not victory_for(board, '0'):
        make_list_of_free_fields(board)
        draw_move(board)
    else:
        print(victory_for(board, '0'))


def make_list_of_free_fields(board):
    return [x for lst in board for x in lst if x not in '0X']


def victory_for(board, sign):
    if sign == 'X':
        winner = 'Computer'
    if sign == '0':
        winner = 'You'
    if set(board[0]) == {sign} or set(board[1]) == {sign} or set(board[2]) == {sign}:
        return f'{winner} won!'
    elif any(set(x) == {sign} for x in ([board[i][j] for i in range(3)] for j in range(3))):
        return f'{winner} won!'
    elif {board[0][0], board[1][1], board[2][2]} == {sign}:
        return f'{winner} won!'
    elif {board[0][2], board[1][1], board[2][0]} == {sign}:
        return f'{winner} won!'
    elif not make_list_of_free_fields(board):
        return 'Tie!'
    else:
        return None


def draw_move(board):
    while True:
        computer_move = str(randrange(1, 10))
        if computer_move in make_list_of_free_fields(board):
            break

    for i in range(3):
        for j in range(3):
            if board[i][j] == computer_move:
                board[i][j] = 'X'

    display_board(board)

    if not victory_for(board, 'X'):
        make_list_of_free_fields(board)
        enter_move(board)
    else:
        print(victory_for(board, 'X'))


display_board(initial)
enter_move(initial)

