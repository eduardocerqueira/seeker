#date: 2023-11-17T16:32:22Z
#url: https://api.github.com/gists/726ef6503aeb372c9a69c16d5ca0d1fe
#owner: https://api.github.com/users/joshika39

board = [
        [7, 8, 0, 4, 0, 0, 1, 2, 0],
        [6, 0, 0, 0, 7, 5, 0, 0, 9],
        [0, 0, 0, 6, 0, 1, 0, 7, 8],
        [0, 0, 7, 0, 4, 0, 2, 6, 0],
        [0, 0, 1, 0, 5, 0, 9, 3, 0],
        [9, 0, 4, 0, 6, 0, 0, 0, 5],
        [0, 7, 0, 3, 0, 0, 0, 1, 2],
        [1, 2, 0, 0, 0, 7, 4, 0, 0],
        [0, 4, 9, 2, 0, 6, 0, 0, 7]
    ]
    
def print_board(b):
    for i in range(len(b)):
        if i % 3 == 0 and i != 0:
            print("──────┼───────┼───────")
        for j in range(len(b[i])):
            if j % 3 == 0 and j != 0:
                print("│ ", end="")
            if j == 8:
                print(b[i][j])
            else:
                print(f"{b[i][j]} ", end="")
                

def valid(b, num, pos):
    # Check row
    for i in range(len(b[0])):
        if b[pos[0]][i] == num and pos[1] != i:
            return False
    # Check column
    for i in range(len(b)):
        if b[i][pos[1]] == num and pos[0] != i:
            return False
    # Check box
    box_x = pos[1] // 3
    box_y = pos[0] // 3
    for i in range(box_y * 3, box_y * 3 + 3):
        for j in range(box_x * 3, box_x * 3 + 3):
            if b[i][j] == num and (i, j) != pos:
                return False
    return True

def find_empty(b):
    for i in range(len(b)):
        for j in range(len(b[i])):
            if b[i][j] == 0:
                return (i, j)
    return None