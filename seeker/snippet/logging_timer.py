#date: 2022-10-10T17:06:11Z
#url: https://api.github.com/gists/379bf88c63abc2e359b6f6072a5ee7d2
#owner: https://api.github.com/users/jongphago

def timer(func):
    def inner(*args):
        begin = time()
        result = func(*args)
        logging.debug(f"{func.__name__}:{time()-begin:.2e}")
        return result
    return inner
