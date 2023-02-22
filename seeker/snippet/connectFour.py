#date: 2023-02-22T17:00:23Z
#url: https://api.github.com/gists/5b650ef7c8a89b7d655d56f0400458f4
#owner: https://api.github.com/users/Varsha-R

def printBoard(board):
    for row in board:
        print(row)

# Initialize board (empty)
def initBoard(nRows, nCols):
    board = []
    for i in range(0,nRows):
        board.append([' '] * nCols)
    S = [ [], [], [], [], [], [], [] ]
    return board, S

# Move randomly
def move(board, S, piece, positionToMove):
    # positionToMove = random.randint(1, 7)
    print("Position to move: ", positionToMove)
    if len(S[positionToMove - 1]) < 6:
        S[positionToMove - 1].append(piece)
        # print(S)
        board[6-len(S[positionToMove-1])][positionToMove-1] = S[positionToMove-1][-1]
        printBoard(board)
    # else:
    #     move(piece, board, S)
    return board, S

def checkWin(board, S):
    game = False
    nRows = len(board)
    nCols = len(board[0])
    # Horizontal check
    for i in range(0,nRows):
        for j in range(3,nCols):
            if (board[i][j] == board[i][j-1] == board[i][j-2] == board[i][j-3]):
                if board[i][j] == 'C' or board[i][j] == 'U':
                    game = True
                else:
                    continue
            else:
                continue
    
    print("Game after horizontal: ", game)
    # Vertical check
    for i in range(0, nCols):
        for j in range(3, nRows):
            if board[j][i]!='' and board[j][i] == board[j-1][i] == board[j-2][i] == board[j-3][i]:
                if board[j][i] == 'C' or board[j][i] == 'U':
                    game = True
                else:
                    continue
            else:
                continue
    
    print("Game after vertical: ", game)
    # Diagonal check
    for i in range(0, nRows-3):
        for j in range(0, nCols-3):
            if board[i][j] == board[i+1][j+1] == board[i+2][j+2] == board[i+3][j+3]:
                if board[i][j] == 'C' or board[i][j] == 'U':
                    game = True
                else:
                    continue
            elif board[i+3][j] == board[i+2][j+1] == board[i+1][j+2] == board[i][j+3]:
                if board[i+3][j] == 'C' or board[i+3][j] == 'U':
                    game = True
                else:
                    continue
            else:
                continue
    
    return game

board, S = initBoard(6, 7)
board, S = move(board, S, "C", 1)
board, S = move(board, S, "U", 2)
board, S = move(board, S, "C", 2)
board, S = move(board, S, "U", 3)
board, S = move(board, S, "U", 3)
board, S = move(board, S, "C", 3)
board, S = move(board, S, "U", 4)
board, S = move(board, S, "U", 4)
board, S = move(board, S, "U", 4)
board, S = move(board, S, "C", 4)
print(checkWin(board, S))