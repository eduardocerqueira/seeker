#date: 2022-12-06T17:00:46Z
#url: https://api.github.com/gists/ee01c262b513ad1594d7e77b17bfcd47
#owner: https://api.github.com/users/x0rworld

import random
from collections import Counter, defaultdict
from typing import Dict, List
from timeit import timeit

random_nums = [random.randrange(1, 10) for _ in range(1000)]


def count_by_dict(random_nums: List[int]) -> Dict:
    result = {}
    for num in random_nums:
        if num not in result:
            result[num] = 1
        else:
            result[num] += 1
    return result


def count_by_defaultdict(random_nums: List[int]) -> Dict:
    result = defaultdict(int)
    for num in random_nums:
        result[num] += 1
    return result


def count_by_counter(random_nums: List[int]) -> Dict:
    counter = Counter()
    for num in random_nums:
        counter[num] += 1
    return counter


def main():
    print("count_by_dict:",
          timeit("count_by_dict(random_nums)", setup="from __main__ import count_by_dict, random_nums", number=1000))
    print("count_by_defaultdict:",
          timeit("count_by_defaultdict(random_nums)", setup="from __main__ import count_by_defaultdict, random_nums", number=1000))
    print("count_by_counter:",
          timeit("count_by_counter(random_nums)", setup="from __main__ import count_by_counter, random_nums", number=1000))


if __name__ == "__main__":
    main()