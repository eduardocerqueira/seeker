#date: 2022-08-25T17:13:51Z
#url: https://api.github.com/gists/385824c15550d8d934be4c0db11387a0
#owner: https://api.github.com/users/kingbode

'''
source
https://gist.github.com/CodeWithHarry/d83fed6958b7626ef51aa87c2d7130de?fbclid=IwAR27Zij46IBW2VNN1X3pcavktbTUZ0XFpzsKHgMHFnFRHvL_xX7enrd2NmA
'''

# refactored version

def printBoard(boardList):
    # update the tick tac toe board
    line = "║ "
    print(f"╔═══╦═══╦═══╗")
    for i in range(0, 9):
        if (boardList[i][1] != 1):
            boardList[i] = (str(i),None)

        line += boardList[i][0] + " ║ "

        if ( (i+1) % 3 == 0):
            print(line)
            if (i+1) != 9:
                print(f"╠═══╬═══╬═══╣")
            else:
                print(f"╚═══╩═══╩═══╝")
            line = "║ "



def checkWin(boardList):

    wins = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]]

    for win in wins:
        checkList = [boardList[win[0]][1], boardList[win[1]][1], boardList[win[2]][1]]
        player = ''.join(set([boardList[win[0]][0], boardList[win[1]][0], boardList[win[2]][0]]))
        if all(checkList) and (player == 'X' or player == 'O'):
            print(f"{player} Won the match")
            return True
    return False


if __name__ == "__main__":

    boardList = []

    boardList = [("player",str(x)) for x in range(9)]

    players = ["O","X"]
    turn = 1 # 0 for X and 1 for O

    print("Welcome to Tic Tac Toe")
    while (True):
        printBoard(boardList)

        if (turn == 1):
            print("X's Chance")

        else:
            print("O's Chance")
        value = int(input("Please enter a value: "))
        # check if the value is already taken
        if (boardList[value][1] == None):
            boardList[value] = (players[turn],1)
        else:
            print("Value already taken")
            continue

        if (checkWin(boardList)):
            print("Match over")
            printBoard(boardList)
            break

        turn = 1 - turn