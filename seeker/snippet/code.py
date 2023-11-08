#date: 2023-11-08T16:46:17Z
#url: https://api.github.com/gists/d2a85e65d203c28447bd1e69f6fc6084
#owner: https://api.github.com/users/kaziamov

import dis
import profile
import tracemalloc


LIMIT = 100_000


def repeater(func):
    def inner():
        for i in range(LIMIT):
            func()
    return inner



def get_2_values():
    return 455, 677, 788


def get_5_values():
    return 899, 900, 901, 902, 903


@repeater
def return_with_variables():
    first = get_2_values()
    second = get_5_values()
    return first + second

@repeater
def return_without_variables():
    return get_2_values() + get_5_values()




cases = [
    return_with_variables,
    return_without_variables
]

if __name__ == "__main__":

    for case in cases:
        tracemalloc.start()
        case()
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        for stat in top_stats:
            print(stat)
        tracemalloc.stop()
        profiler = profile.Profile()
        profiler.runcall(case)
        profiler.print_stats()
        print(dis.dis(case))
        print("---------------")