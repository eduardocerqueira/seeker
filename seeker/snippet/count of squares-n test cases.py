#date: 2024-01-17T17:07:02Z
#url: https://api.github.com/gists/31af8de061ecb0c7a6fde57752975556
#owner: https://api.github.com/users/NiluferDastan

import math


def count_of_squares(numbers: list[int]) -> int:
    count = 0
    for number in numbers:
        if number == math.sqrt(number) ** 2:
            count += 1
        return count


def main():
    n = int(input())
    for i in range(n):
        numbers = input().split(" ")
        lists = []
        for number in numbers:
            lists.append(int(number))
        print(count_of_squares(lists))


if __name__ == '__main__':
    main()
