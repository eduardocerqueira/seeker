#date: 2023-07-03T16:38:53Z
#url: https://api.github.com/gists/274124e633f7c7dc1770752e8cf43d91
#owner: https://api.github.com/users/ALAxHxC

LIST_OF_ISLANDS_4 = [
    [1, 1, 0, 1],
    [1, 1, 0, 1],
    [0, 0, 0, 0],
    [1, 0, 1, 0]
]

LIST_OF_ISLANDS_3 = [
    [1, 0, 0, 1],
    [1, 0, 0, 1],
    [0, 0, 0, 0],
    [1, 1, 1, 1]
]

LIST_OF_ISLANDS_1 = [
    [0, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
]

LIST_OF_ISLANDS_2 = [
    [0, 0, 0, 0],
    [0, 1, 1, 0],
    [0, 0, 1, 0],
    [1, 0, 0, 0]
]

def is_consecutive_island(x: int, y: int, island: list[list[int]], count=0):
    if island[x][y] == 0:
        return count
    if island[x][y] == 1:
        count = 1
    if x + 1 < len(island) and y + 1 < len(island[0]) and island[x + 1][y + 1] == 1:
        island[x + 1][y + 1] = 2
        is_consecutive_island(x + 1, y + 1, island, count=1)
    if x + 1 < len(island) and island[x + 1][y] == 1:
        island[x + 1][y] = 2
        is_consecutive_island(x + 1, y, island, count=1)
    if y + 1 < len(island) and island[x][y + 1] == 1:
        island[x][y + 1] = 2
        is_consecutive_island(x, y + 1, island, count=1)
    return count


def validate_is_island(island: list[list[int]]):
    count = 0
    x = 0
    while x < len(island):
        y = 0
        while y < len(island):
            count = count + is_consecutive_island(x, y, island, count=0)
            y = y + 1

        x = x + 1
    return count


if __name__ == '__main__':
    n_islands = validate_is_island(LIST_OF_ISLANDS_4)
    assert n_islands == 4, f"Should be 3, but it is {n_islands}"
    n_islands = validate_is_island(LIST_OF_ISLANDS_3)
    assert n_islands == 3, f"Should be 3, but it is {n_islands}"
    n_islands = validate_is_island(LIST_OF_ISLANDS_1)
    assert n_islands == 1, f"Should be 1, but it is {n_islands}"
    n_islands = validate_is_island(LIST_OF_ISLANDS_2)
    assert n_islands == 2, f"Should be 2, but it is {n_islands}"
    print("Everything OK, well done!")
