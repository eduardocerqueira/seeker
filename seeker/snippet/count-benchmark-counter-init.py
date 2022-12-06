#date: 2022-12-06T17:09:14Z
#url: https://api.github.com/gists/6c17b09db89f44f42c8780d4805a57e6
#owner: https://api.github.com/users/x0rworld

import random
from collections import Counter, defaultdict
from typing import Dict, List
from timeit import timeit

random_nums = [random.randrange(1, 10) for _ in range(1000)]

# ignore the other counting functions

def count_by_counter_init(random_nums: List[int]) -> Dict:
    counter = Counter(random_nums)
    return counter


def main():
    print("count_by_dict:",
          timeit("count_by_dict(random_nums)", setup="from __main__ import count_by_dict, random_nums", number=1000))
    print("count_by_defaultdict:",
          timeit("count_by_defaultdict(random_nums)", setup="from __main__ import count_by_defaultdict, random_nums", number=1000))
    print("count_by_counter:",
          timeit("count_by_counter(random_nums)", setup="from __main__ import count_by_counter, random_nums", number=1000))
    print("counter_by_counter_init:",
          timeit("count_by_counter_init(random_nums)", setup="from __main__ import count_by_counter_init, random_nums", number=1000))


if __name__ == "__main__":
    main()
