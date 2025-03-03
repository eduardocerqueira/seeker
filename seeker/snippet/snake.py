#date: 2025-03-03T16:54:27Z
#url: https://api.github.com/gists/66aeb53797d3eb4b555b11832ccd1b79
#owner: https://api.github.com/users/ThomasRohde

# /// script
# description = "Simulate and display Conway's Game of Life in the console using the Textual package"
# authors = [
#   'Script-Magic AI Generator',
# ]
# date = '2023-10-05'
# requires-python = '>=3.9'
# dependencies = [
#   'textual>=0.1.18',
# ]
# tags = [
#   'generated',
#   'script-magic',
# ]
# ///
# Generated from the prompt: "Game of life in the console using Textual package"
import random
import argparse
from textual.app import App
from textual.widgets import Static
from textual.viewport import Viewport

# Define the size of the grid
def default_grid(cols, rows):
    return [[random.choice([False, True]) for _ in range(cols)] for _ in range(rows)]

def next_generation(grid):
    rows, cols = len(grid), len(grid[0])
    new_grid = [[False for _ in range(cols)] for _ in range(rows)]
    for y in range(rows):
        for x in range(cols):
            # Compute the number of alive neighbours
            alive_neighbors = 0
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < rows and 0 <= nx < cols:
                        alive_neighbors += grid[ny][nx]
            # Apply the rules of Game of Life
            if grid[y][x]:
                if alive_neighbors in [2, 3]:
                    new_grid[y][x] = True
            else:
                if alive_neighbors == 3:
                    new_grid[y][x] = True
    return new_grid

# Custom Textual Widget to display the grid
class GameOfLifeWidget(Static):
    def __init__(self, grid, **kwargs):
        super().__init__(**kwargs)
        self.grid = grid

    def render(self):
        state = ""
        for row in self.grid:
            state += ''.join(['█' if cell else ' ' for cell in row]) + "\n"
        return state

    def update_grid(self):
        self.grid = next_generation(self.grid)
        self.refresh()

# Textual App
class GameOfLifeApp(App):
    CSS_PATH = ''
    BINDINGS = [
        ("q", "quit", "Quit"),
    ]

    async def on_mount(self):
        # Fetching command line arguments
        parser = argparse.ArgumentParser(description='Run Conway\'s Game of Life in the console using Textual.')
        parser.add_argument('--cols', type=int, default=40, help='Number of columns')
        parser.add_argument('--rows', type=int, default=20, help='Number of rows')
        args = parser.parse_args()

        grid = default_grid(args.cols, args.rows)
        widget = GameOfLifeWidget(grid)
        self.view.dock(widget, edge="top", size=args.rows)
        self.set_interval(0.5, widget.update_grid)

# Main entry point
if __name__ == "__main__":
    GameOfLifeApp.run()