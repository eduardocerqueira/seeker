#date: 2025-05-02T16:35:44Z
#url: https://api.github.com/gists/398cbf011fc9f06ebbe36d45e6b2f655
#owner: https://api.github.com/users/NotSebbbb


#Name: Sebastian Donahue
#Date: 5/2/25
#Course Number: 113
#Course Name: Intro to python
#Problem Number: chapter 8 project 1
#Email: sldonahue2201@student.stcc.edu
#Problem Description: Create a fully functioning Tic Tac Toe game.


# **********************************************
# imports here


# **********************************************
# Define as many constants or variables you need here

TITLE = "Tic Tac Toe V1.0" #title
CONTINUE_PROMPT = "Do this again? [y/N] " #prompt to continue

# **********************************************
# Define as many functions you need here

def printBoard(board):
    print("   | " + board[0][1] + " |   ")
    print("---+---+---")
    print(" " + board[1][0] + " | " + board[1][1] + " | " + board[1][2])
    print("---+---+---")
    print("   | " + board[2][1] + " |   ")

def getTurns(board):
    return str(sum(cell != ' ' for row in board for cell in row))

def getLocation(board, player):
    while True:
        try:
            move = input(f"Player {player}, enter row and column (1-3 space-separated): ").strip()
            if not move:
                print("Input cannot be empty.")
                continue
            parts = move.split()
            if len(parts) != 2:
                print("Please enter exactly two numbers.")
                continue
            row, col = int(parts[0]) - 1, int(parts[1]) - 1
            if not (0 <= row < 3 and 0 <= col < 3):
                print("Coordinates must be between 1 and 3.")
                continue
            if not isValid(board, row, col):
                print("That space is already taken.")
                continue
            return row, col
        except ValueError:
            print("Invalid input. Please enter numeric values.")

def isValid(board, row, col):
    return board[row][col] == ' '

def playMove(board, player, row, col):
    board[row][col] = player

def isWinner(board, p):
    for i in range(3):
        if all(board[i][j] == p for j in range(3)):  # Check rows
            return True
        if all(board[j][i] == p for j in range(3)):  # Check columns
            return True
    if all(board[i][i] == p for i in range(3)):      # Main diagonal
        return True
    if all(board[i][2 - i] == p for i in range(3)):  # Anti-diagonal
        return True
    return False

def isFull(board):
    return all(cell != ' ' for row in board for cell in row)

# **********************************************
# Start your logic coding in the process function
def process():
    board = [[' ' for _ in range(3)] for _ in range(3)]
    player = 'X'

    while True:
        printBoard(board)
        row, col = getLocation(board, player)
        playMove(board, player, row, col)
        if isWinner(board, player) or isFull(board):
            break
        player = 'O' if player == 'X' else 'X'

    printBoard(board)
    if isWinner(board, 'X'):
        status = "X is the winner!"
    elif isWinner(board, 'O'):
        status = "O is the winner!"
    else:
        status = "The game is a tie."

    status += " After " + getTurns(board) + " plays."
    print(status)

# **********************************************
# Do not change the do_this_again function
def do_this_again(prompt):
    do_over = input(prompt)
    return do_over.strip().lower() == 'y'

# **********************************************
# Do not change the main function
def main():
    print(f"Welcome to {TITLE}")
    while True:
        process()
        if not do_this_again(CONTINUE_PROMPT):
            break
    print(f"Thank you for using {TITLE}")

if __name__ == "__main__":
    main()