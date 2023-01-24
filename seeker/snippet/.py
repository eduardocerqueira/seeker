#date: 2023-01-24T16:47:31Z
#url: https://api.github.com/gists/000e7cf899829450cab381a8291cdae8
#owner: https://api.github.com/users/nicoKoehler

def maxValue(board, d=0):

    value = -2
    actionValues = []

    if terminal(board):
        print("TERMINAL FOUND!!!!")
        return utility(board), actionValues, d

    print("MAX called", end=" ")
    for a in actions(board):
        print(a)
        min_val = minValue(result(board, a),d = d+1)
        value = max(value, min_val[0])

        # recommended action will always take first in action array. so optimal solutions are inserted ad pos 0, draw-solutions at the end. 
        if value == 1: actionValues.insert(0,[a, min_val[2]])
        elif value == 0: actionValues.append(a)

    return value, actionValues, d


def minValue(board,d=0):
    
    value = 2
    actionValues = []

    if terminal(board):
        return utility(board), actionValues, d
    
    print("MIN Called", end=" ")
    for a in actions(board):
        print(a)
        min_val = maxValue(result(board, a), d = d+1)
        value = min(value, min_val[0])
        
        # recommended action will always take first in action array. so optimal solutions are inserted ad pos 0, draw-solutions at the end. 
        if value == -1: actionValues.insert(0,[a,min_val[2]])
        elif value == 0: actionValues.append(a)

    return value, actionValues, d
