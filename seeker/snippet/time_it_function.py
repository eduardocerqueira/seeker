#date: 2023-02-06T17:03:23Z
#url: https://api.github.com/gists/f0323f5ead0cde6892485d414e6f0a1f
#owner: https://api.github.com/users/jgrugru

from typing import Callable, List
import time
import statistics


def time_my_function(
    fn: Callable, n: int = 100, precision: int = 6, output: bool = True, *args, **kwargs
) -> List[float]:
    """
    Timing function\n
    Provides mean, median, stdev, min, and max time for fn call. Runs the given function n times, default 100.
    Returns a list of all run times List[float].
    """
    list_of_times = []
    for x in range(n):
        start = time.time()
        fn(*args, **kwargs)
        end = time.time()
        list_of_times.append(end - start)

    average = round(sum(list_of_times) / n, precision)
    std_dev = round(statistics.stdev(list_of_times), precision)
    minimum = round(min(list_of_times), precision)
    maximum = round(max(list_of_times), precision)

    if output:
        print(f"{n} times: mean={average}, median={round(statistics.median(list_of_times), precision)}, stdev={std_dev}, min={minimum}, max={maximum}")

    return list_of_times
