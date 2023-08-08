#date: 2023-08-08T17:00:26Z
#url: https://api.github.com/gists/87b1a94a9f53fdaf9e07ca7be5d1b1c8
#owner: https://api.github.com/users/amandagloor

from random import randrange

board = [[1, 2, 3], [4, 'X', 6], [7, 8, 9]]

def display_board(board):
    # The function accepts one parameter containing the board's current status
    # and prints it out to the console.
    print("+-------+-------+-------+")
    for row in board: 
        print("|       |       |       |")
        print(f"|   {row[0]}   |   {row[1]}   |   {row[2]}   |")
        print("|       |       |       |")
        print("+-------+-------+-------+")
        
def enter_move(board):
    # The function accepts the board's current status, asks the user about their move, 
    # checks the input, and updates the board according to the user's decision.
    while True:
        move = input("Enter Move: ")    # ask player for input
        if move.isdigit() and 1 <= int(move) <= 9:    # check for valid input
            row = (int(move) - 1) // 3    # Find input row, column
            col = (int(move) - 1) % 3
            if board[row][col] == 'O' or  board[row][col] == 'X':    # valid move check
                print("Invalid move. Please try again.")
            else:
                board[row][col] = 'O'    # insert move
                break
        else: 
            print("Invalid move. Please try again.")

def make_list_of_free_fields(board):
    # The function browses the board and builds a list of all the free squares; 
    # the list consists of tuples, while each tuple is a pair of row and column numbers.
    free_fields = []    # create list of available moves
    for i in range(3):
        for j in range(3):
            if board[i][j] != 'O' and board[i][j] != 'X':    # add available moves to list
                free_fields.append((i, j))
    return free_fields

def victory_for(board, sign):
    # The function analyzes the board's status in order to check if 
    # the player using 'O's or 'X's has won the game  
    for row in board:   # Check rows
        if row[0] == row[1] == row[2] == sign:
            return True
    for col in range(3):    # Check columns
        if board[0][col] == board[1][col] == board[2][col] == sign:
            return True
    if board[0][0] == board[1][1] == board[2][2] == sign:   # Check diagonals
        return True
    if board[0][2] == board[1][1] == board[2][0] == sign:
        return True

def tie(board):
    free_fields = make_list_of_free_fields(board)   # Check for a tie
    if not free_fields:
        return True
    return False

def draw_move(board):
    # The function draws the computer's move and updates the board. 
    free_fields = make_list_of_free_fields(board)    # create updated free fields list
    if free_fields: 
        index = randrange(len(free_fields))    # generate random index number to choose move from list
        row, col = free_fields[index]
        board[row][col] = 'X'    # insert move

while True:
    display_board(board)    # console displays board with available moves for player
    
    enter_move(board)    # prompts player for a move
    
    if victory_for(board, 'O'):    # checks if player won
        display_board(board)
        print("You won!")
        break    # Ends game when player wins
       
    draw_move(board)    # CPU takes its turn 

    if tie(board):
        display_board(board)
        print("It's a tie!")    # checks for a tie
        break

    if victory_for(board, 'X'):    # checks if CPU won
        display_board(board)
        print("The computer won!")
        break # ends game when CPU wins