#date: 2023-12-11T16:45:51Z
#url: https://api.github.com/gists/b6842b05eadc8524b752d894a346d7e1
#owner: https://api.github.com/users/brass75

GALAXY = '#'


def parse_input(input_: str) -> list[list[str]]:
    return [list(line) for line in input_.splitlines()]


def min_distance(start: tuple[int, int], dest: tuple[int, int],
                 rows: list[int], cols: list[int], expansion: int) -> int:
    x1, y1 = start
    x2, y2 = dest
    row_range = range(min(x1, x2) + 1, max(x1, x2))
    col_range = range(min(y1, y2) + 1, max(y1, y2))
    mult = sum(n in row_range for n in rows) * (expansion - 1) + sum(n in col_range for n in cols) * (expansion - 1)

    return abs(x1 - x2) + abs(y1 - y2) + mult


def expand(input_: list[list[str]]) -> tuple[list[int], list[int]]:
    return ([i for i, row in enumerate(input_) if '#' not in row],
            [i for i in range(len(input_[0])) if not any(row[i] == GALAXY for row in input_)])


def get_galactic_coordinates(map_: list[list[str]]) -> list[tuple[int, int]]:
    return [(i, j) for i, row in enumerate(map_)
            for j, c in enumerate(row)
            if c == GALAXY]


def solve(input_: str, expansion: int = 2) -> int:
    if not input_:
        return 0
    parsed = parse_input(input_)
    coordinates = get_galactic_coordinates(parsed)
    expanded = expand(parsed)
    pairs = ({tuple(sorted((start, dest)))
              for start in coordinates
              for dest in coordinates
              if start != dest})
    return sum(min_distance(*pair, *expanded, expansion=expansion) for pair in pairs)


if __name__ == '__main__':
    expected = [(374, [2])]
    for i, e in enumerate(expected):
        e_total, e_params = e
        assert (total := solve(TEST_INPUT, *e_params)) == e_total, f'Test {1} for part 1 failed! {total=} {e_total=}'
        print(f'Part 1: [test {i}] {total}')
    total = solve(INPUT)
    print(f'Part 1: {total}')

    expected = [(1030, [10]), (8410, [100])]
    for i, e in enumerate(expected):
        e_total, e_params = e
        assert (total := solve(TEST_INPUT, *e_params)) == e_total, f'Test {1} for part 2 failed! {total=} {e_total=}'
        print(f'Part 2: [test {i}] {total}')
    total = solve(INPUT, 1000000)
    print(f'Part 2: {total}')
