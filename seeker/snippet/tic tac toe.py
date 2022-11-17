#date: 2022-11-17T17:08:30Z
#url: https://api.github.com/gists/8143d36f152a8199b87c68991a949d3b
#owner: https://api.github.com/users/baronswe

game_board = ['-', '-', '-',
              '-', '-', '-',
              '-', '-', '-', ]

winner = None

game_in_progress = True

current_player = 'X'


def display_game_board():
    print(f'{game_board[0]} | {game_board[1]} | {game_board[2]}')
    print(f'{game_board[3]} | {game_board[4]} | {game_board[5]}')
    print(f'{game_board[6]} | {game_board[7]} | {game_board[8]}')


def main():
    display_game_board()
    # continue until condition becomes false
    while game_in_progress:

        players_input(current_player)

        check_for_winner()

        check_for_tie()

        flip_player()

    if winner == 'X' or winner == 'O':
        print(f'{winner} won')
    else:
        print('tie.')


def players_input(player):
    # Get position from player
    print(f"{player}'s turn.")
    users_input = input("Choose a position from 1-9: ")

    # Make sure the users input is valid
    valid = False
    while not valid:

        while users_input not in ["1", "2", "3", "4", "5", "6", "7", "8", "9"]:
            users_input = input("invalid input. Choose a position from 1-9: ")

        # Get correct index in the game board list
        users_input = int(users_input) - 1

        # make sure the spot is available on the game board
        if game_board[users_input] == "-":
            valid = True
        else:
            print("You can't go there. Go again.")

    # Put the game piece on the game board
    game_board[users_input] = player

    display_game_board()


def check_for_winner():
    # global variable
    global winner
    # check if there was a winner
    row_winner = check_rows()

    column_winner = check_columns()

    diagonal_winner = check_diagonals()

    # locate the winner
    if row_winner:
        winner = row_winner

    elif column_winner:
        winner = column_winner

    elif diagonal_winner:
        winner = diagonal_winner

    else:
        winner = None


def check_for_tie():
    # global variable
    global game_in_progress
    # If board is full
    if "-" not in game_board:
        game_in_progress = False
        return True
    # Else there is no tie
    else:
        return False


def check_rows():
    # global variable
    global game_in_progress
    # Check if any of the rows have all the same value (and is not empty)
    row_1 = game_board[0] == game_board[1] == game_board[2] != "-"
    row_2 = game_board[3] == game_board[4] == game_board[5] != "-"
    row_3 = game_board[6] == game_board[7] == game_board[8] != "-"
    # If any row does have a match, there is a win
    if row_1 or row_2 or row_3:
        game_in_progress = False
    # Return the winner
    if row_1:
        return game_board[0]
    elif row_2:
        return game_board[3]
    elif row_3:
        return game_board[6]
        # Or return None if there was no winner
    else:
        return None


def check_columns():
    # global variables
    global game_in_progress
    # Check if any of the columns have all the same value (and is not empty)
    column_1 = game_board[0] == game_board[3] == game_board[6] != "-"
    column_2 = game_board[1] == game_board[4] == game_board[7] != "-"
    column_3 = game_board[2] == game_board[5] == game_board[8] != "-"
    # If any row does have a match, there is a win
    if column_1 or column_2 or column_3:
        game_in_progress = False
    # Return the winner
    if column_1:
        return game_board[0]
    elif column_2:
        return game_board[1]
    elif column_3:
        return game_board[2]
        # Or return None if there was no winner
    else:
        return None


def check_diagonals():
    # global variable
    global game_in_progress
    # Check if any of the columns have all the same value (and is not empty)
    diagonal_1 = game_board[0] == game_board[4] == game_board[8] != "-"
    diagonal_2 = game_board[2] == game_board[4] == game_board[6] != "-"
    # If any row does have a match, there is a win
    if diagonal_1 or diagonal_2:
        game_in_progress = False
    # Return the winner
    if diagonal_1:
        return game_board[0]
    elif diagonal_2:
        return game_board[2]
    # Or return None if there was no winner
    else:
        return None


def flip_player():
    # Global variable
    global current_player
    # If the current player was X, make it O
    if current_player == "X":
        current_player = "O"
    # Or if the current player was O, make it X
    elif current_player == "O":
        current_player = "X"


# call main
main()
