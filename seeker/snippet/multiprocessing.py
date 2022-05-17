#date: 2022-05-17T16:57:22Z
#url: https://api.github.com/gists/b2b546e223c1b66e09c89950eb359b21
#owner: https://api.github.com/users/searope

import random
import time
import multiprocessing as mp
import itertools as it

# just an example generator to prove lazy access by printing when it generates
def get_counter(limit=10):
    for i in range(limit):
        for j in range(limit):
            print(f"YIELDED: {i},{j}")
            yield i, j

# a utility function to get us a slice of an iterator, as an iterator
# when working with iterators maximum lazyness is preferred 
def iterator_slice(iterator, length):
    iterator = iter(iterator)
    while True:
        res = tuple(it.islice(iterator, length))
        if not res:
            break
        yield res[0]

# our process function, just prints what's passed to it and waits for 1-6 seconds
def test_process(values):
    i, j = values
    time_to_wait = random.random() * 5
    #print(f"START: {i},{j}, waiting: {time_to_wait:0.2f} seconds")
    time.sleep(time_to_wait)
    print(f"END: {i},{j}")
    return i*j

if __name__ == "__main__":
    with mp.Pool(processes=2) as pool:  # lets use just 2 workers
        count = get_counter(4)  # get our counter iterator set to iterate from 0-29
        count_iterator = iterator_slice(count, 1)  # we'll process them in chunks of 7
        queue = []  # a queue for our current worker async results, a deque would be faster
        while count_iterator or queue:
            try:
                # add our next slice to the pool:
                queue.append(pool.apply_async(test_process, [next(count_iterator)]))
            except (StopIteration, TypeError):  # no more data, clear out the slice iterator
                count_iterator = None
            # wait for a free worker or until all remaining workers finish
            while queue and (len(queue) >= pool._processes or not count_iterator):
                process = queue.pop(0)  # grab a process response from the top
                process.wait(0.1)  # let it breathe a little, 100ms should be enough
                if not process.ready():  # a sub-process has not finished execution
                    queue.append(process)  # add it back to the queue
                else:
                    # you can use process.get() to get the result if needed
                    #print(f"RESULT: {process.get()}")  # print the result
                    pass
