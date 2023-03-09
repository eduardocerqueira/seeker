#date: 2023-03-09T17:09:34Z
#url: https://api.github.com/gists/0cdf016d1bae7e18b79d7170d47cb21c
#owner: https://api.github.com/users/milov52

# Задача: B.Ловкость рук
# id успешной попытки 83450273
from typing import Dict, List, Tuple

MATRIX_SIZE = 4


def get_button_map(matrix: List[List[str]]) -> Dict:
    button_map = {}
    for i in range(MATRIX_SIZE):
        for j in range(MATRIX_SIZE):
            if matrix[i][j] != '.':
                if matrix[i][j] not in button_map:
                    button_map[matrix[i][j]] = 1
                else:
                    button_map[matrix[i][j]] += 1
    return button_map


def get_score(k: int, matrix: List[List[str]]) -> int:
    button_map = get_button_map(matrix)
    return sum(val <= 2 * k for val in button_map.values())


def read_input() -> Tuple[int, List[List[str]]]:
    k = int(input())
    matrix = []

    for i in range(MATRIX_SIZE):
        matrix.append(input())
    return k, matrix


if __name__ == '__main__':
    k, matrix = read_input()
    print(get_score(k, matrix))
