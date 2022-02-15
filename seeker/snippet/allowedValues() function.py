#date: 2022-02-15T17:07:01Z
#url: https://api.github.com/gists/ff4d7331dab080c3d003c1478d53f990
#owner: https://api.github.com/users/evangelos-zafeiratos

def allowedValues(board,row,col):

    numbersList = list()

    for number in range(1,10):

        found = False
        # Check if all row elements include this number
        for j in range(9):
            if board[row][j] == number:
                found = True
                break
        # Check if all column elements include this number
        if found == True:
            continue
        else:
            for i in range(9):
                if board[i][col] == number:
                    found = True
                    break

        # Check if the number is already included in the block
        if found == True:
            continue
        else:
            rowBlockStart = 3* (row // 3)
            colBlockStart = 3* (col // 3)

            rowBlockEnd = rowBlockStart + 3
            colBlockEnd = colBlockStart + 3
            for i in range(rowBlockStart, rowBlockEnd):
                for j in range(colBlockStart, colBlockEnd):
                    if board[i][j] == number:
                        found = True
                        break
        if found == False:
            numbersList.append(number)
    return numbersList
