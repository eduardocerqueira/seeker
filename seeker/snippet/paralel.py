#date: 2024-11-12T17:05:18Z
#url: https://api.github.com/gists/aef6745a434ccbca3c7ab47d771e43cd
#owner: https://api.github.com/users/caio-pavesi

"""Example of parallel execution using multiprocessing"""

import time
import inspect
from multiprocessing import Queue, Process

def main() -> None:
    """Here we use the multiprocessing library to
    execute functions in parallel. Since both functions
    have a return value, we use the Queue class to get
    the results.

    The expected execution time is 5 seconds, as both
    functions have a sleep of 5 seconds (actually 5.1 -
    5.2 seconds)."""

    start = time.time()

    return_ = Queue()  # To get the return values of the functions

    f1 = Process(target=funcao_1, args=(return_,))
    f2 = Process(target=funcao_2)
    f3 = Process(target=funcao_3, args=(return_,))

    return_ = executar_em_paralelo([f1, f2, f3], return_)

    # Get the results
    print(f"results {return_}")

    print(f"finished in {time.time() - start}")

def executar_em_paralelo(
    processos: list[Process],
    queue: Queue = None
) -> dict[str, any] | None:
    """Executes processes in parallel and returns the results
    in a dictionary with each function name that executed.

    Args:
        processos (list[Process]): List of processes to be executed.
        queue (Queue, optional): Queue to get the results from the functions. Defaults to None.

    Returns:
        dict[str, any] | None: Dictionary with function names and their results, or None if no queue is provided.
    """

    # Start the processes in parallel
    for processo in processos:
        processo.start()

    # Wait for the processes to finish
    for processo in processos:
        processo.join()

    if not queue:
        return None

    results = dict()
    for _ in range(queue.qsize()):
        try:
            func_name, return_ = queue.get()
            results[func_name] = return_

        except TypeError as exception:
            raise TypeError("One of the functions running in parallel is not using the 'resultado_paralelo' function to return the result") from exception

    return results

def resultado_paralelo(queue: Queue, return_: any) -> None:
    """Returns the result of a function.

    Args:
        queue (Queue): Queue to return the result of the function.
        return_ (any): Result of the function.
    """

    caller_func_name = inspect.stack()[1][3]

    queue.put((caller_func_name, return_))

def funcao_1(return_: Queue) -> int:
    """Any function.

    Args:
        return_ (Queue): Queue to return the result of the function.

    Returns:
        int: Result of the function.
    """

    print(f"Executing function 1 at: {time.time()}")
    time.sleep(1)

    resultado_paralelo(return_, 1)
    return 1

def funcao_2():
    """Any function."""

    print(f"Executing function 2 at: {time.time()}")
    time.sleep(5)

def funcao_3(return_: Queue) -> int:
    """Any function.

    Args:
        return_ (Queue): Queue to return the result of the function.

    Returns:
        int: Result of the function.
    """

    print(f"Executing function 3 at: {time.time()}")
    time.sleep(5)

    resultado_paralelo(return_, 3)
    return 3

if __name__ == "__main__":
    main()