#date: 2022-09-09T17:14:54Z
#url: https://api.github.com/gists/12466950913b2dc4e7025928a6a943df
#owner: https://api.github.com/users/McCarthyCode

#!/bin/python3
# maze.py

import sys
from random import sample


class Maze:
    """
    Class for generating and displaying a maze based on width and height parameters
    """

    ##############
    # Properties #
    ##############

    # Include carriage return character in string output line endings.
    CR = False

    # Include carriage return character in string output line endings.
    AXIS_LABELS = False

    # Default coordinates of start and end locations
    START, END = (0, 0), (-1, -1)

    # Maximum recursion depth
    MAX_RECURSION_LIMIT = 1500

    @property
    def __str__(self) -> str:
        """
        Property method returning a text-based graphical output of the maze
        """

        # Compile a list of strings, line-by-line, to be joined with the
        # appropriate line endings.
        str_list = []
        if self.AXIS_LABELS:
            str_list += [
                '', ''.join('      %2d' % x for x in range(self.width))
            ]
        str_list.append(
            ('    +' if self.AXIS_LABELS else '+') + '-------+' * self.width)

        for row in range(self.height):
            # Define vertical walls w/ cell values
            center = ' %2d |' % row if self.AXIS_LABELS else '|'
            for col in range(self.width):
                center += '   %s   %s' % (
                    self.grid[row][col], '|' if self.has_ewall(
                        (row, col)) else ' ')

            # Define vertical walls w/o cell values
            vwalls = '    |' if self.AXIS_LABELS else '|'
            for col in range(self.width):
                vwalls += '       '
                vwalls += '|' if self.has_ewall((row, col)) else ' '

            # Define horizontal walls
            hwalls = '    +' if self.AXIS_LABELS else '+'
            for col in range(self.width):
                hwalls += '%s+' % (
                    str('-' * 7) if self.has_swall(
                        (row, col)) else str(' ' * 7))

            # Put the individual row lines together
            str_list.append(vwalls)
            str_list.append(center)
            str_list.append(vwalls)
            str_list.append(hwalls)

        # Compile the list into one string joined by line endings.
        ret = ''
        if self.CR:
            ret = '\r\n'.join(str_list)
        else:
            ret = '\n'.join(str_list)

        return ret

    ###########
    # Methods #
    ###########

    def __init__(
            self,
            width: int,
            height: int,
            start: tuple[int, int] = START,
            end: tuple[int, int] = END) -> None:
        """
        Main entry point for the maze object
        """

        # Set width and height with minimum values of 2 each.
        self.width = max(width, 2)
        self.height = max(height, 2)
        self.size = self.width * self.height

        sys.setrecursionlimit(min(self.MAX_RECURSION_LIMIT, self.size))

        # Set coordinates for starting and ending locations
        self.start = (start[0] % height, start[1] % width)
        self.end = end[0] % height, end[1] % width

        # Mark start and end positions and fill empty cells with spaces.
        self.grid = [[' ' for _ in range(width)] for _ in range(height)]
        self.grid[start[0]][start[1]] = 'S'
        self.grid[end[0]][end[1]] = 'E'

        # Define relationships between cells.
        self.generate()

    def print(self) -> None:
        """
        Display the maze in standard output.
        """

        print(self.__str__)
        print(
            'The maze can be solved in %d step%s.' %
            (self.steps, '' if self.steps == 1 else 's'),
            file=sys.stderr)

    def has_ewall(self, pos: tuple[int, int]) -> bool:
        """
        Boolean method returning whether or not a cell has a wall to the East
        """

        # Define/unpack tuples
        row, col = pos
        east = row, col + 1

        # DO display a wall if the cell borders the maze's easternmost edge,
        # or if the cell does is not yet related to any other cells.
        #
        # Otherwise, a wall IS NOT displayed.
        return not (
            pos in self.relations and east in self.relations[pos]
            and east in self.relations and pos in self.relations[east])

    def has_swall(self, pos: tuple[int, int]) -> bool:
        """
        Boolean method returning whether or not a cell has a wall to the South
        """

        # Define/unpack tuples
        row, col = pos
        south = row + 1, col

        # DO NOT display a southern wall under any of the following conditions:
        #
        # a) the cell is the end cell, or
        # b) a relation exists between cells in question.
        #
        # Otherwise, a wall IS displayed.
        return not (
            pos in self.relations and south in self.relations[pos]
            and south in self.relations and pos in self.relations[south])

    def generate(self) -> None:
        """
        Helper method for initializing the relations dict and generating paths
        via DFS
        """

        self.relations = {}
        self.steps = self.generate_rec()

    def generate_rec(self, pos: tuple[int, int] = None, depth: int = 0) -> int:
        """
        Method to randomly choose paths, filling the grid relations dict in a
        tree-like graph structure which:

        a) includes a path from start to end,
        b) has no cycles,
        c) fills the entire grid, and
        d) is constructed with DFS

        Returns the shortest path to the exit
        """
        # Specify default starting location
        if not pos:
            pos = self.start

        # Return if current position is the end.
        if pos == self.end:
            return depth

        # Unpack tuple parameter and define neighboring positions.
        row, col = pos
        options = (
            (row, col + 1), (row - 1, col), (row, col - 1), (row + 1, col))

        # Track shortest path to exit
        shortest_path = self.size

        # Randomize neighbors and iterate through and adding relationships to
        # unvisited cells.
        for opt in sample(options, 4):
            # Ignore already visited neighbors.
            if opt in self.relations:
                continue

            # Ignore out-of-bounds coordinates.
            y, x = opt
            if x in range(self.width) and y in range(self.height):

                # Define flags for doubly linked verticies.
                forward, backward = False, False
                if not opt in self.relations or not pos in self.relations[opt]:
                    forward = True

                if not pos in self.relations or not opt in self.relations[pos]:
                    backward = True

                # Add the relations Based on flags determined *before* doing so.
                if forward:
                    try:
                        self.relations[pos].add(opt)
                    except KeyError:
                        self.relations[pos] = {opt}

                if backward:
                    try:
                        self.relations[opt].add(pos)
                    except KeyError:
                        self.relations[opt] = {pos}

                #  Recursive call; keep trailblazin' (yee-haw)
                shortest_path = min(
                    shortest_path, self.generate_rec(opt, depth + 1))

        return shortest_path


if __name__ == '__main__':
    """
    Main entry point for the script (if run directly)
    """

    # m = Maze(79, 33)  # fullscreen (font 4)
    # m = Maze(37, 31)  # windowed (font 4)

    # m = Maze(23, 12)  # fullscreen
    m = Maze(11, 11, (0, 0), (5, 5))  # windowed

    m.print()
