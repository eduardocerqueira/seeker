#date: 2026-01-08T17:16:37Z
#url: https://api.github.com/gists/defacee184f5c2418bfd1c3a7e7634e8
#owner: https://api.github.com/users/dcronin05

import os
import random
import time

# Taken from: https://www.geeksforgeeks.org/dsa/program-for-conways-game-of-life/
# Date accessed: 2026-01-08
# Description: Python method to find the next generation
# of a given matrix of cells
def findNextGen(mat):
    m, n = len(mat), len(mat[0])

    # create matrix to store cells of next generation.
    nextGen = [[0 for _ in range(n)] for _ in range(m)]

    # Directions of eight possible neighbors
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0),
                  (1, 1), (-1, -1), (1, -1), (-1, 1)]

    # Iterate over the matrix
    for i in range(m):
        for j in range(n):
            live = 0

            # Count the number of live neighbors
            for dx, dy in directions:
                x, y = i + dx, j + dy

                # Check if the neighbor is live
                if 0 <= x < m and 0 <= y < n and (mat[x][y] == 1):
                    live += 1

            # If current cell is live and number of live neighbors
            # is less than 2 or greater than 3, then the cell will die
            if mat[i][j] == 1 and (live < 2 or live > 3):
                nextGen[i][j] = 0

            # If current cell is dead and number of live neighbors
            # is equal to 3, then the cell will become live
            elif mat[i][j] == 0 and live == 3:
                nextGen[i][j] = 1

            # else the state of cells
            # will remain same.
            else:
                nextGen[i][j] = mat[i][j]

    for i in range(m):
        for j in range(n):
            print(nextGen[i][j], end=" ")
        print()

    return nextGen


if __name__ == '__main__':

    os.environ['TERM'] = 'xterm'
    os.system('clear')

    mat = [[random.randint(0, 1) for _ in range(40)] for _ in range(40)]

    for i in range(500):
        mat = findNextGen(mat)
        i += 1
        time.sleep(.1)
        os.system('clear')
