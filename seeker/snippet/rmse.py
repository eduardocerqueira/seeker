#date: 2023-08-15T17:00:30Z
#url: https://api.github.com/gists/708dc94949baf190a8b37efd3aa69527
#owner: https://api.github.com/users/bewagner

import math


def solution(predicted, observed):
    if len(predicted) != len(observed):
        raise RuntimeError("Sequences had different length")
    n = len(predicted)
    squared_differences = [(p - o) ** 2 for p, o in zip(predicted, observed)]
    summed_differences = sum(squared_differences) / n
    return math.sqrt(summed_differences)


if __name__ == '__main__':
    predicted = [4, 25, 0.75, 11]
    observed = [3, 21, -1.25, 13]
    print(solution(predicted, observed))