#date: 2022-06-15T16:58:36Z
#url: https://api.github.com/gists/898964c7f37b3f872bcfcaba1ab603d5
#owner: https://api.github.com/users/stephanie-wang

import ray


@ray.remote
def map(start, end, boundaries):
    vals = list(range(start, end))
    partitions = []

    prev_bound = 0
    for next_bound in boundaries:
        partitions.append([x for x in vals if x >= prev_bound and x < next_bound])
    partitions.append([x for x in vals if x >= prev_bound])
    return partitions

@ray.remote
def reduce(*map_results):
    map_results = [x for y in map_results for x in y]
    return sorted(map_results)

@ray.remote
def reduce_generator(multiple_map_results):
    for map_results in multiple_map_results:
        # NOTE: Now we're passing a doubly nested list of ObjectRefs and we
        # don't pass them as direct args, so we have to call ray.get here to
        # get the actual values.
        map_results = ray.get(map_results)
        map_results = [x for y in map_results for x in y]
        yield sorted(map_results)


if __name__ == "__main__":
    # 1 reduce output per task.
    num_map = 10
    num_reduce = 2

    boundaries = [50]
    map_results = [
            map.options(num_returns=len(boundaries) + 1).remote(0, 100, boundaries)
            for _ in range(num_map)]
    reduce_results = []
    for i in range(num_reduce):
        reduce_results.append(reduce.remote(*[map_result[i] for map_result in map_results]))

    # Same number of reduce tasks, but multiple outputs per task.
    num_map = 10
    num_reduce = 10
    num_reduce_tasks = 2

    boundaries = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    map_results = [map.options(num_returns=len(boundaries) + 1).remote(0, 100, boundaries)
            for _ in range(num_map)]
    reduce_results = []
    num_reduce_returns = num_reduce // num_reduce_tasks
    for i in range(num_reduce_tasks):
        reduce_args = []
        for j in range(num_reduce_returns):
            reduce_args.append([map_result[i * num_reduce_returns + j] for map_result in map_results])

        reduce_results += reduce_generator.options(num_returns=num_reduce_returns).remote(reduce_args)