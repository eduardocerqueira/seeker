#date: 2023-02-22T16:54:44Z
#url: https://api.github.com/gists/36f7288e67a77eb9e11699c9c894124d
#owner: https://api.github.com/users/idram1984

class CustomMultiprocessing:
    """
    Class to process function using multiprocessing. The function needs to return a dataFrame.
    """
    DEFAULT_NCPU = multiprocessing.cpu_count()

    def __init__(self, **kwargs):
        self.__n_cpu = kwargs.get('num_cpu', multiprocessing.cpu_count())

    def exec_group_in_parallel(self, tab_parameter, func, logger=sys.stdout) -> pd.DataFrame:
        start = time.time()
        logger.flush()
        logger.write("\nUsing {} CPUs in parallel ... \n.".format(self.__n_cpu))

        with multiprocessing.Pool(self.__n_cpu) as pool:
            result = pool.starmap_async(func, tab_parameter)
            cycler = itertools.cycle('\|/-')

            while not result.ready():
                value = "\rTasks left: {}. {}\t".format(result._number_left, next(cycler))
                logger.write(value)
                logger.flush()
                time.sleep(0.1)
            got = result.get()

        logger.write("\nTasks completed. Processed {} groups in {:.1f}s\n".format(
            len(got), time.time() - start))

        return pd.concat(got)