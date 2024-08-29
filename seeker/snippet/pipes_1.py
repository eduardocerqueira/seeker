#date: 2024-08-29T17:03:30Z
#url: https://api.github.com/gists/f52bd8bf0c1090729519f5f426bf2264
#owner: https://api.github.com/users/officialdun

from collections import deque

def main(filename):
    grid = {}
    source = None

    with open(filename) as file:
        for line in file:
            char, x, y = line.strip().split()
            x, y = int(x), int(y)
            if char == '*':
                source = (x, y)
            grid[(x, y)] = char

    # BFS setup
    queue = deque([source])
    visited = set([source])
    connected = set()

    # Directions for adjacent cells
    directions = [(0, 1, '║╝╚╩╣╠╬*', '║╗╔╦╣╠╬', 'up'),  # Going up
                  (1, 0, '═╔╚╦╩╠╬*', '═╗╝╣╩╦╬', 'right'),  # Going right
                  (0, -1, '║╗╔╦╣╠╬*', '║╝╚╩╣╠╬', 'down'), # Going down
                  (-1, 0, '═╗╝╣╩╦╬*', '═╔╚╦╩╠╬','left')] # Going left
    
    # BFS traversal
    while queue:
        x, y = queue.popleft()
        for dx, dy, cFrom, cTo, direction in directions:
            nx, ny = x + dx, y + dy
            if (nx, ny) in visited:
                continue
            if (nx, ny) in grid:
                fromCell = grid[(x,y)]
                cell = grid[(nx, ny)]
                doesConnect = fromCell in cFrom and (cell.isalpha() or cell in cTo)
                # print(f'{fromCell} → {cell} going {direction}. connected: {doesConnect}')
                if cell == '*' or doesConnect:
                    if cell.isalpha():
                        connected.add(cell)
                    queue.append((nx, ny))
                    visited.add((nx, ny))

    return ''.join(sorted(connected))

print(main('coding_qual_input.txt'))
