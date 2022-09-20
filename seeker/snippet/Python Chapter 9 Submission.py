#date: 2022-09-20T17:21:04Z
#url: https://api.github.com/gists/dce6a6bdc4fa5f498d2401499a5354af
#owner: https://api.github.com/users/Tilly-Pielichaty

import random
import json
import os

def main():
    show_header()
    player1, player2 = get_players()

    #create the board
    # Board is a list of rows
    # Rows are a list of cells
    show_leaderboard()
    board = [
        [None, None, None],
        [None, None, None],
        [None, None, None],
        ]

    #Choose initial player
    active_player_index = 0
    #players = ["Ben", "Computer"]
    symbols = ["X","O"]

    #Until someone wins
    while not find_winner(board):
        #show the board
        player = players[active_player_index]
        symbol = symbols[active_player_index]

        announce_turn(player)
        show_board(board)
        if choose_location(board, symbol):
            print("That isn't an option, try again.")
            continue

        # Toggle active player
        active_player_index = (active_player_index + 1) % 2

def show_header():
    print("--------------------")
    print(" Knoughts and Crosses")
    print(" Leaderboard edition")
    print("--------------------")

def get_players():
    p1 = input("Player 1, what is your name? ")
    p2 = "Computer"

    return p1, p2

def show_leaderboard():
    leaders = load_leaders()

    sorted_leaders = list(leaders.items())
    sorted_leaders.sort(key=lambda l: l[1], reverse=True)

    print()
    print("LEADERS:")
    for name, wins in sorted_leaders[0:5]:
        print(f"{wins:,} --- {name}")
    print()
    print("--------------------")
    print()


def choose_location(board, symbol):
    row = int(input("Choose which row: "))
    column = int(input("Choose which column: "))

    row -= 1
    column -= 1
    if row < 0 or row >= len(board):
        return False
    if column < 0 or column >= len(board[0]):
        return False

    cell = board[row][column]
    if cell is not None:
        return False

    board[row][column] = symbol
    return True

def show_board(board):
    for row in board:
        print(" | ", end="")
        for cell in row:
            symbol = cell if cell is not None else "_"
            print(symbol, end=" | ")
        print()


def announce_turn(player):
    print()
    print(f"It's {player}'s turn. Here's the board: ")
    print()

def play_game(player_1, player_2):
    log(f"New game starting between {player_1} and {player_2}.")
    wins = {player_1: 0, player_2: 0}

    winner = find_winner(board)
    if winner is None:
        msg = "This round was a tie!"
        print(msg)
        log(msg)
    else:
        msg = f"{winner} takes the round!"
        print(msg)
        log(msg)
        wins[winner] += 1

        # print(f"Current win status : {wins}")

        msg = f"Score is {player_1}: {wins[player_1]} and {player_2}: {wins[player_2]}."
        print(msg)
        log(msg)
        print()

    overall_winner = find_winner(wins, wins.keys())
    msg = f"{overall_winner} wins the game!"
    print(msg)
    log(msg)
    record_win(overall_winner)

def load_leaders():
    directory = os.path.dirname(__file__)
    filename = os.path.join(directory, "leaderboard2.json")

    if not os.path.exists(filename):
        return {}

    with open(filename, "r", encoding="utf-8") as fin:
        return json.load(fin)

def record_win(winner_name):
    leaders = load_leaders()

    if winner_name in leaders:
        leaders[winner_name] += 1
    else:
        leaders[winner_name] = 1

    directory = os.path.dirname(__file__)
    filename = os.path.join(directory, "leaderboard2.json")

    with open(filename, "w", encoding="utf-8") as fout:
        json.dump(leaders, fout)

def log(msg):
    directory = os.path.dirname(__file__)
    filename = os.path.join(directory, "rps.log")
    with open(filename, "a", encoding= "utf-8") as fout:
        import datetime
        fout.write(f"[{datetime.datetime.now().date().isoformat()}] ")
        fout.write(msg)
        fout.write("\n")

def find_winner(board):

    rows = board
    for row in rows:
        symbol1 = rows[0]
        if symbol1 and all(symbol1 == cell for cell in rows):
            return True

    columns = []
    for col_idx in range(0,3):
        col = [
            board[0][col_idx],
            board[1][col_idx],
            board[2][col_idx],
        ]
        columns.append(col)

    for col in columns:
        symbol1 = col[0]
        if symbol1 and all(symbol1 == cell for cell in col):
            return True

    diagonals = [
        [board[0][0]. board[1][1], board[2][2]],
        [board[0][2].board[1][1], board[2][0]],
    ]

    for diag in diagonals:
        symbol1 = diag[0]
        if symbol1 and all(symbol1 == cell for cell in diag):
            return True

    return False

if __name__=='__main__':
    main()
