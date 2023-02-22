#date: 2023-02-22T16:59:23Z
#url: https://api.github.com/gists/e07283e43ddf828d67e6d7e605a3b0df
#owner: https://api.github.com/users/Varsha-R

def printBoard(board):
    for row in board:
        print(row)

def createBoard(n):
    board = []
    for row in range(n):
        rowValues = []
        for col in range(n):
            if row % 2 == 0 and col % 2 == 0 or row % 2 != 0 and col % 2 != 0:
                rowValues.append(1)
            elif row % 2 == 0 and col % 2 != 0 or row % 2 != 0 and col % 2 == 0:
                rowValues.append(0)
        board.append(rowValues)
    printBoard(board)
    return board

directions_1 = [(1, -1), (1, 1)]
directions_2 = [(-1, -1), (-1, 1)]

def computeValidMoves(board, rowVal, colVal, playerPositions, player):
    validMoves = []
    directions = directions_1 if player == '1' else directions_2
    opponent = "1" if player == "2" else "2"
    
    for direction in directions:
        x, y = direction
        if 0<=(rowVal+x)<8 and 0<=(colVal+y)<8 and board[rowVal+x][colVal+y] == 0 and (rowVal+x, colVal+y) not in playerPositions[player]:
                if (rowVal+x, colVal+y) in playerPositions[opponent]: 
                    if (rowVal+2*x, colVal+2*y) not in playerPositions[opponent]:
                        validMoves.append((rowVal+2*x, colVal+2*y))
                else:
                    validMoves.append((rowVal+x, colVal+y))
    return validMoves
    
def move(player, playerPositions, fromPosition, toPosition, validMoves):
    # print(validMoves, fromPosition)
    if toPosition in validMoves[fromPosition]:
        indexToReplace = playerPositions[player].index(fromPosition)
        playerPositions[player][indexToReplace] = toPosition
    return playerPositions
        

board = createBoard(8)
playerPositions = {
    "1": [(0, 1), (0, 3), (0, 5), (0, 7), (1, 0), (1, 2), (1, 4), (1, 6), (2, 1), (2, 3), (2, 5), (2, 7)],
    "2": [(5, 0), (5, 2), (5, 4), (5, 6), (6, 1), (6, 3), (6, 5), (6, 7), (7, 0), (7, 2), (7, 4), (7, 6)]
}

validMoves = {}
for player in playerPositions.keys():
    print("All valid moves of player " + player + ":")
    validMoves[player] = {}
    for position in playerPositions[player]:
        validMoves[player][position] = computeValidMoves(board, position[0], position[1], playerPositions, player)
        print("{} -> {}".format(position, validMoves[player][position]))

playerPositions = move("2", playerPositions, (5, 4), (4, 3), validMoves["2"])
playerPositions = move("1", playerPositions, (2, 5), (3, 4), validMoves["1"])
print(playerPositions)

for player in playerPositions.keys():
    print("All valid moves of player: " + player)
    for position in playerPositions[player]:
        allValidMoves = computeValidMoves(board, position[0], position[1], playerPositions, player)
        print("{} -> {}".format(position, allValidMoves))