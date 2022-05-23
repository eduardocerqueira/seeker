#date: 2022-05-23T17:14:20Z
#url: https://api.github.com/gists/e997b72d0e8f5b74407184369d0e2b22
#owner: https://api.github.com/users/ramazanemreosmanoglu

import pyglet
import random
import typing
from pyglet.window import key


SIZE = (600, 700)
BOARD_SIZE = 12

COLORS = [
    (128,0,128), # Purple
    (255,255,0), # Yellow
    (255,0,0), # Red
    (128,128,128), # White
    (255,0,255), # Pink
    (0,255,0), # Green
]

TITLE = "Drench"

DEFAULT_HEIGHT = 32
DEFAULT_WIDTH = 32

LEFT = 'left'
DOWN = 'down'

def is_valid_pos(pos):
    """
    Checks whether the given position is valid or not.

    - Positions shouldn't contain numbers lower than 0.
    """

    for i in pos:
        if i < 0:
            return False

    return True

def create_block(row, col, color, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT, board_size=BOARD_SIZE, batch=None):
    bwidth = board_size*width
    bheight = board_size*height
    constant_height = 100

    first_block_pos = (
        (SIZE[0]/2)-(bwidth/2),
        (SIZE[1]/2)-(bheight/2)+constant_height,
    )


    block_pos = (
        first_block_pos[0] + (width*(row)),
        first_block_pos[1] + (height*(col)),
    )


    return pyglet.shapes.Rectangle(block_pos[0], block_pos[1], width, height, color=color, batch=batch)


class DrenchGame(pyglet.window.Window):
    def __init__(self):
        super().__init__(SIZE[0], SIZE[1], TITLE)

        self.batch = pyglet.graphics.Batch()
        self.board = self._create_board()
        self.visited = set()

    def _create_board(self):
        random_color = lambda: random.choice(COLORS)
        board = []

        for i in range(BOARD_SIZE):
            row = []

            for j in range(BOARD_SIZE):
                row.append(
                    create_block(j, i, random_color(), batch=self.batch),
                )

            board.append(row)

        return board

    def on_draw(self):
        self.clear()
        self.batch.draw()

    def on_mouse_press(self, x, y, button, modifier):
        print(f'({x}, {y})')

    def on_key_press(self, symbol, modifiers):
        if symbol == key.R:
            self.paint((11,0), COLORS[2])

        if symbol == key.Y:
            self.paint((11,0), COLORS[1])

        if symbol == key.W:
            self.paint((11,0), COLORS[3])

        if symbol == key.P:
            self.paint((11,0), COLORS[0])

        if symbol == key.K:
            self.paint((11,0), COLORS[4])

        if symbol == key.G:
            self.paint((11,0), COLORS[5])

    def _paint_util(self, pos, color):
        """
        Pos: (x, y): Current position to start painting.
        """

        if pos in self.visited:
            return
        self.visited.add(pos)

        main_block = self.board[pos[0]][pos[1]]
        old_color = main_block.color
        main_block.color = color

        def check(target_pos):
            try:
                target = self.board[target_pos[0]][target_pos[1]]
            except IndexError:
                pass
            else:
                if tuple(target.color) == tuple(main_block.color) or tuple(target.color) == tuple(old_color):
                    self._paint_util(target_pos, main_block.color)

        # Right
        check((pos[0], pos[1]+1))

        # Down
        check((pos[0]-1, pos[1]))

    def paint(self, pos, color):
        self.visited.clear()
        self._paint_util(pos,color)


if __name__ == "__main__":
    win = DrenchGame()
    pyglet.app.run()
