#date: 2024-12-06T16:59:52Z
#url: https://api.github.com/gists/4f7cf83808d775c3b3c6829d94eda8bb
#owner: https://api.github.com/users/gxjakkap

def pkm():
    h = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    cur = input() # get input. e.g.: e4
    cur = list(cur) # split each character to list. e.g. [e, 4]
    curInt = [h.index(cur[0].upper()) + 1, int(cur[1])] # change horizontal axis to int (+1 since index is 0 based)
    allPossibleMove = []

    # two horizontal + one vertical
    for i in [2, -2]:
        x = curInt[0] + i # get horizontal axis move
        if (x > 0) and (x < 9): # check if said move is possible/not off the board
            for j in [1, -1]: 
                y = curInt[1] + j # get vertical move 
                if (y > 0) and (y < 9): # check
                    allPossibleMove.append(f'{h[x - 1]}{y}') # append the move to list, change first character back to letter using 'h' list (-1 for 0 based index)
    
    # two vertical + one horizontal (do the same thing as above but swap horizontal/vertical)
    for i in [1, -1]:
        x = curInt[0] + i
        if (x > 0) and (x < 9):
            for j in [2, -2]:
                y = curInt[1] + j
                if (y > 0) and (y < 9):
                    allPossibleMove.append(f'{h[x - 1]}{y}')
    
    allPossibleMove.sort() # sort list alphanumerically
    
    for i in allPossibleMove:
        print(i)


pkm()