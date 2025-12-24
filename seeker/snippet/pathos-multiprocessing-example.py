#date: 2025-12-24T17:10:01Z
#url: https://api.github.com/gists/e0eb2d5c0650710399190727086b0f8d
#owner: https://api.github.com/users/sgouda0412

from pathos.multiprocessing import ProcessingPool as Pool

def parallel_map(func, array, n_workers):
    def compute_batch(i):
        try:
            return func(i)
        except KeyboardInterrupt:
            raise RuntimeError("Keyboard interrupt")

    p = Pool(n_workers)
    err = None
    # pylint: disable=W0703,E0702
    # some bs boilerplate from StackOverflow
    try:
        return p.map(compute_batch, array)
    except KeyboardInterrupt, e:
        print 'got ^C while pool mapping, terminating the pool'
        p.terminate()
        err = e
    except Exception, e:
        print 'got exception: %r:, terminating the pool' % (e,)
        p.terminate()
        err = e

    if err is not None:
        raise err

# example
if __name__ == '__main__':
    import time
    def do_work(i):
        print('italian strike %i' % i)
        time.sleep(2)
        return i

    # note: can't pickle functions with the normal multiprocessing package
    print(parallel_map(do_work, range(10), n_workers=3))