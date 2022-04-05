#date: 2022-04-05T17:15:32Z
#url: https://api.github.com/gists/b6a51d4b01f8150c52ed0015415df904
#owner: https://api.github.com/users/Alex-Alen

import random as rnd


def check_right(x: int, y: int, map: [[int]]) -> bool:
    if y + 1 >= len(map[0]):
        return False
    if map[x][y + 1] == 1:
        return False
    else:
        return True


def check_left(x: int, y: int, map: [[int]]) -> bool:
    if y - 1 <= 0:
        return False
    if map[x][y - 1] == 1:
        return False
    else:
        return True


def check_up(x: int, y: int, map: [[int]]) -> bool:
    if x - 1 <= 0:
        return False
    if map[x - 1][y] == 1:
        return False
    else:
        return True


def check_down(x: int, y: int, map: [[int]]) -> bool:
    if x + 1 >= len(map[0]):
        return False
    if map[x + 1][y] == 1:
        return False
    else:
        return True


if __name__ == '__main__':
    # создать начальные координаты и карту
    x = 0
    y = 0
    map = [
        [12, 0, 1, 1, 1],
        [1, 0, 1, 0, 1],
        [0, 0, 0, 0, 0],
        [0, 1, 1, 0, 1],
        [1, 0, 0, 0, 24]
    ]

    while map[x][y] != 24:
        free: [(int, int)] = []

        # какие клетки доступны
        if check_right(x, y, map):
            free.append((x, y + 1))
        if check_left(x, y, map):
            free.append((x, y - 1))
        if check_up(x, y, map):
            free.append((x - 1, y))
        if check_down(x, y, map):
            free.append((x + 1, y))

        # рандомно выбираю клетку
        random_hod_iz_vazmoznih = free[rnd.randint(1, len(free)) - 1]

        # меняю координаты на выбронаю клетку (x, y)
        x = random_hod_iz_vazmoznih[0]
        y = random_hod_iz_vazmoznih[1]

    print("URAAAA FINISH!!!", x, y)
