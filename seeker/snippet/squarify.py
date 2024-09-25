#date: 2024-09-25T17:04:08Z
#url: https://api.github.com/gists/38ee52a398e41abaef04812a9422cfd9
#owner: https://api.github.com/users/fourpoints

"""Treemap Squarify Algortihm

This is based on the algorithm defined in the following paper:
https://www.win.tue.nl/~vanwijk/stm.pdf -- 3.2 Algorithm

It is intended to be as close to the definition as possible.
As such it defies many laws of good standards,
such as having global variables and using recursion.

It is separated into two parts: Layout and squarify.
The layout part defines the positions and sizes of
the smaller rectangles, while the squarify part defines
the subdivision algorithm.

There are a few mistakes in the paper:
* The inequality seems to be wrong.
  (Possible sign error on the worst function.)
* The worst and head functions are ill-defined.
  (They are undefined for empty sequence.)
* No stopping condition.

Either fractions.Fraction or Fraction=operator.truediv can be used.

A requirement is that `sum(rectangle areas) == large rectangle area.
"""

import math
# from fractions import Fraction
from operator import truediv as Fraction

# --- Layout algorithm

rectangles = []


def rect(x, y, width, height):
    return locals()


def place_rectangles(row):
    x = Rectangle.width
    y = Rectangle.height
    s = sum(row)
    if Rectangle.width > Rectangle.height:
        w = sum(row) / Rectangle.height
        for a in row:
            h = a/w
            rectangles.append(rect(-x, -y, w, h))
            y = y - h
    else:
        h = sum(row) / Rectangle.width
        for a in row:
            w = a/h
            rectangles.append(rect(-x, -y, w, h))
            x = x - w


# --- Squarify algorithm

class Rectangle:
    width = Fraction(6, 1)
    height = Fraction(4, 1)


def layout_row(row):
    # Place rectangles (optional)
    place_rectangles(row)

    # Shrink rectangle
    if Rectangle.width > Rectangle.height:
        Rectangle.width = Rectangle.width - sum(row) / Rectangle.height
    else:
        Rectangle.height = Rectangle.height - sum(row) / Rectangle.width


def get_width():
    return min(Rectangle.width, Rectangle.height)


# def worst(R, w):
#     (Almost) equivalent to the below function
#     return max(max(w*w*r/s*s, s*s/(w*w*r)) for r in R)

def worst(R, w):
    if len(R) == 0:
        return float("inf")

    s = sum(R)
    return max(
        Fraction(w**2 * max(R), s**2),
        Fraction(s**2, w**2 * min(R)))


def squarify(children, row, w):
    if len(children) == len(row) == 0:
        return

    c = children[:1]
    if worst(row, w) > worst(row + c, w):
        squarify(children[1:], row + c, w)
    else:
        layout_row(row)
        squarify(children, [], get_width())


def main():
    areas = [6, 6, 4, 3, 2, 2, 1]
    
    squarify(areas, [], get_width())
    
    print(rectangles)


if __name__ "__main__":
    main()
