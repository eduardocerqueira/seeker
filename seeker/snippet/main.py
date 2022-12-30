#date: 2022-12-30T16:50:15Z
#url: https://api.github.com/gists/ac128dcc0b9b70de065d1d4f9f3b87d4
#owner: https://api.github.com/users/dennisseah

from math import floor

# example of puzzle9x9_1.txt (0 = value to fill)
#
# 0 4 3 0 8 0 2 5 0
# 6 0 0 0 0 0 0 0 0
# 0 0 0 0 0 1 0 9 4
# 9 0 0 0 0 4 0 7 0
# 0 0 0 6 0 8 0 0 0
# 0 1 0 2 0 0 0 0 3
# 8 2 0 5 0 0 0 0 0
# 0 0 0 0 0 0 0 0 5
# 0 3 4 0 9 0 7 1 0

# strategy
# 1. locate all the 0 and maintain a matrix of their possible values
#    this involves getting the values in the bounding_box, row and col
#    in the above example for row 0 and col 0, it would be [1, 7], the rest
#    of the numbers are taken

#    0 4 3 0 8 0 2 5 0
#    6 0 0
#    0 0 0
#    9
#    0
#    0
#    8
#    0
#    0
#
# 2. with the possible values matrix, get the array that is the minimum length
#    and iterate through the values of this array to assign value to the puzzle.
# 3. repeat step #2 until all the values are filled. We need to backtrack when we
#    find that the puzzle is not completed (having 0s) and there are no possible
#    values for the cell with 0 value.
# 4. There are also cases when we use up all the possible values and we cannot
#    complete the puzzle. Then there are no solutions to the puzzle.


def bounding_box(row, col):
    top = floor(row / 3) * 3
    left = floor(col / 3) * 3
    rect = []
    for i in range(top, top+3):
        rect.extend([[i, j] for j in range(left, left+3)])
    return rect


def already_assigned_values(puzz, row, col):
    vals = set([i for i in puzz[row]])

    for idx in range(0, len(puzz[row])):
        vals.add(puzz[idx][col])

    for r in bounding_box(row, col):
        vals.add(puzz[r[0]][r[1]])
    return vals


def get_choices(puzz, row, col):
    return [n for n in [x+1 for x in range(len(puzz[0]))] if n not in already_assigned_values(puzz, row, col)]


def get_all_choices(puzz):
    choices = []

    for row in range(0, len(puzz)):
        row_choices = []
        for col in range(0, len(puzz[row])):
            row_choices.append(get_choices(puzz, row, col)
                               if puzz[row][col] == 0 else None)
        choices.append(row_choices)

    return choices


def next_item(puzz, choices):
    next_item = None

    for idx, row in enumerate(choices):
        for col in range(0, len(row)):
            if puzz[idx][col] == 0 and \
                    row[col] is not None and \
                    len(row[col]) > 0 and \
                    (next_item is None or len(row[col]) < len(choices[next_item[0]][next_item[1]])):
                next_item = (idx, col)
    return next_item


def remove_choices(choices, val, row, col):
    tracker = []

    def remove(row, col):
        cell = choices[row][col]
        if cell is not None and val in cell:
            cell.remove(val)
            tracker.append((row, col))

    for i, _ in enumerate(choices[row]):
        remove(row, i)

    for i in range(len(choices[0])):
        remove(i, col)

    for r in bounding_box(row, col):
        remove(r[0], r[1])

    return tracker


def backtrack(choices, val, tracker):
    for c in tracker:
        choices[c[0]][c[1]].append(val)


def is_valid(puzz, choices, row, col):
    def check(r, c):
        choice = choices[r][c]
        cell = puzz[r][c]
        return cell != 0 or (choice is not None and len(choice) > 0)

    for i, cell in enumerate(puzz[row]):
        if not check(row, i):
            return False

    for i in range(9):
        if not check(i, col):
            return False

    for r in bounding_box(row, col):
        if not check(r[0], r[1]):
            return False

    return True


def solve(puzz, choices):
    item = next_item(puzz, choices)

    if not item:
        return True

    row, col = item[0], item[1]
    vals = choices[row][col]

    for val in vals:
        tracker = remove_choices(choices, val, row, col)
        puzz[row][col] = val
        if is_valid(puzz, choices, row, col):
            if solve(puzz, choices):
                return True
            backtrack(choices, val, tracker)
            puzz[row][col] = 0
        else:
            backtrack(choices, val, tracker)
            puzz[row][col] = 0

    return False


if __name__ == "__main__":
    with open("puzzle9x9_1.txt") as f:
        puzzle = []

        for line in f.readlines():
            puzzle.append([int(x) for x in line.split(" ")])

        choices = get_all_choices(puzzle)
        if solve(puzzle, choices):
            for p in puzzle:
                print(p)
        else:
            print("there are no solutions")
