#date: 2022-10-20T17:36:58Z
#url: https://api.github.com/gists/90dcb1b83af2faa3b407172521f1824b
#owner: https://api.github.com/users/aleenprd

def timing(f: Callable) -> None:
    """Times a function runtime in minutes.

    Args:
        f (callable): a function/method.
    """
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        exec_time = round((te - ts) / 60, 4)
        print(f"\nExecution time: {exec_time} minutes.")

    return wrap