#date: 2025-12-03T17:06:31Z
#url: https://api.github.com/gists/1b2e928cd0a73d6ab8ce9fae4051189c
#owner: https://api.github.com/users/muhammad64759-dev

# Intelligent Search-Driven Evacuation System
# Implements manual BFS and DFS on 3 custom grids
# Author: Your Name
# Date: 2025-12-03

from collections import deque
import copy

# -------------------------------
# Cell types
# -------------------------------
FREE = 0
BLOCKED = 1
HAZARD = 2
START = 'S'
EXIT = 'E'

# Movement directions: Up, Down, Left, Right
directions = [(-1,0), (1,0), (0,-1), (0,1)]

# -------------------------------
# Node class for BFS and DFS
# -------------------------------
class Node:
    def __init__(self, r, c, parent=None):
        self.r = r
        self.c = c
        self.parent = parent

# -------------------------------
# Function to print grid with optional path
# -------------------------------
def print_grid(grid, path=None):
    display = copy.deepcopy(grid)
    if path:
        for r, c in path:
            if display[r][c] not in (START, EXIT):
                display[r][c] = '*'
    for row in display:
        print(' '.join(str(cell) for cell in row))
    print()

# -------------------------------
# BFS Implementation
# -------------------------------
def bfs(grid, start, exit_points):
    visited = set()
    queue = deque([Node(start[0], start[1])])
    visited.add(start)
    visited_order = []

    while queue:
        current = queue.popleft()
        visited_order.append((current.r, current.c))

        if (current.r, current.c) in exit_points:
            # Reconstruct path
            path = []
            node = current
            while node:
                path.append((node.r, node.c))
                node = node.parent
            path.reverse()
            return path, visited_order

        for dr, dc in directions:
            nr, nc = current.r + dr, current.c + dc
            if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]):
                if (nr, nc) not in visited and grid[nr][nc] != BLOCKED:
                    queue.append(Node(nr, nc, current))
                    visited.add((nr, nc))

    return None, visited_order

# -------------------------------
# DFS Implementation
# -------------------------------
def dfs(grid, start, exit_points):
    visited = set()
    stack = [Node(start[0], start[1])]
    visited.add(start)
    visited_order = []

    while stack:
        current = stack.pop()
        visited_order.append((current.r, current.c))

        if (current.r, current.c) in exit_points:
            # Reconstruct path
            path = []
            node = current
            while node:
                path.append((node.r, node.c))
                node = node.parent
            path.reverse()
            return path, visited_order

        for dr, dc in directions:
            nr, nc = current.r + dr, current.c + dc
            if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]):
                if (nr, nc) not in visited and grid[nr][nc] != BLOCKED:
                    stack.append(Node(nr, nc, current))
                    visited.add((nr, nc))

    return None, visited_order

# -------------------------------
# Function to run a test case
# -------------------------------
def run_test(grid, start, exit_points, algorithm='BFS'):
    print("Grid Layout:")
    print_grid(grid)

    if algorithm == 'BFS':
        path, visited_order = bfs(grid, start, exit_points)
    else:
        path, visited_order = dfs(grid, start, exit_points)

    print(f"{algorithm} Visited Order:")
    print(visited_order)

    if path:
        print(f"{algorithm} Path Found (length {len(path)-1}):")
        print(path)
        print("Grid with Path:")
        print_grid(grid, path)
    else:
        print(f"{algorithm} could not find a path.")
    print("="*60)

# -------------------------------
# Custom Test Grids
# -------------------------------

# Grid 1
grid1 = [
    [FREE, FREE, BLOCKED, FREE, EXIT],
    [FREE, BLOCKED, FREE, BLOCKED, FREE],
    [START, FREE, FREE, FREE, FREE],
    [BLOCKED, FREE, BLOCKED, FREE, BLOCKED],
    [FREE, FREE, FREE, FREE, FREE]
]
start1 = (2,0); exit1 = [(0,4)]

# Grid 2
grid2 = [
    [START, BLOCKED, FREE, FREE, FREE],
    [FREE, BLOCKED, BLOCKED, FREE, EXIT],
    [FREE, FREE, FREE, BLOCKED, FREE],
    [BLOCKED, FREE, FREE, FREE, FREE],
    [FREE, FREE, BLOCKED, FREE, FREE]
]
start2 = (0,0); exit2 = [(1,4)]

# Grid 3 with hazards
grid3 = [
    [START, FREE, HAZARD, BLOCKED, EXIT],
    [BLOCKED, HAZARD, FREE, HAZARD, FREE],
    [FREE, FREE, BLOCKED, FREE, FREE],
    [HAZARD, BLOCKED, FREE, FREE, HAZARD],
    [FREE, FREE, FREE, BLOCKED, FREE]
]
start3 = (0,0); exit3 = [(0,4)]

# -------------------------------
# Run all tests
# -------------------------------
print("=== Test Grid 1 ===")
run_test(grid1, start1, exit1, 'BFS')
run_test(grid1, start1, exit1, 'DFS')

print("=== Test Grid 2 ===")
run_test(grid2, start2, exit2, 'BFS')
run_test(grid2, start2, exit2, 'DFS')

print("=== Test Grid 3 ===")
run_test(grid3, start3, exit3, 'BFS')
run_test(grid3, start3, exit3, 'DFS')
