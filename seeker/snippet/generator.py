#date: 2023-03-29T17:41:51Z
#url: https://api.github.com/gists/b4a1bf25fd1af5eb3c9e6e579fbd9a66
#owner: https://api.github.com/users/jelc53

def sqrt(a: float) -> Iterator[float]:
    """
    With this version, we update x at each iteration and then yield the updated value. In-
    stead of getting a single value, the caller of the function gets an iterator that contains an
    infinite number of iterations; it is up to the caller to decide how many iterations to evalu-
    ate and when to stop.
    """
    x = a / 2 # initial guess
    while True:
        x = (x + (a / x)) / 2
        yield x

def converge(values: Iterator[float], threshold: float) -> Iterator[float]:
    """
    Since we now have a first-class abstraction for iteration, we can write a general-purpose converge function that takes an iterator
    and returns a version of that same iterator that stops as soon as two values are suﬀiciently close.
    """
    for a, b in itertools.pairwise(values):
        yield a
        
        if abs(a - b) < threshold:
            break

if __name__ == '__main__':
    results = converge(sqrt(n), 0.001)
    capped_results = itertools.islice(results, 10000)