#date: 2025-07-08T17:00:02Z
#url: https://api.github.com/gists/f60b932eaeaea48f85e3a7de7acae8fd
#owner: https://api.github.com/users/pete-otaqui


def retry(
    *,
    max_attempts: int = 3,
    calculate_wait_time: Callable[[int], float] = lambda attempt: round(
        random.uniform(0, attempt) + attempt**2, 2
    ),
    retry_on_exception: Callable[[Exception], bool] = lambda e: True,
    retry_message: Optional[
        Callable[[Exception, int, float, tuple, dict], None]
    ] = lambda e, attempt, wait_time, *args, **kwargs: print(
        f"Failed attempt {attempt}, retrying after {wait_time} seconds"
    ),
) -> Callable:
    """
    Retry a function multiple times if it raises and exception.

    :param max_attempts: The maximum number of attempts to make, defaults to 3.
    :param calculate_wait_time: Return the time to wait after failing an attempt number, defaults to the number squared plus some jitter.
    :param retry_on_exception: A function that returns True if the exception should be retried, defaults to always retry.
    :param retry_message: A function that prints a message when retrying, defaults to printing a message.

    ========
    Examples
    ========

    Simple retry example:

    .. code-block:: python

        @retry()
        def my_function():
            print("Hello")
            raise Exception("This is an error")

        my_function()

    Custom retry message, using original parameter values:

    .. code-block:: python

        def custom_retry_message(e, attempt, wait_time, *args, **kwargs):
            # refer to the original parameters if you like:
            print(f"Failed saying hello to {args[0]}, retrying in {wait_time} seconds")

        @retry(retry_message=custom_retry_message)
        def my_function(name: str):
            raise Exception("This is an error")
            print(f"Hello {name}")

    For brevity you can just pass in lambdas too:

    .. code-block:: python

        @retry(
            max_attempts=5, # try 5 times
            retry_message=lambda : return None, # no message
            calculate_wait_time=lambda attempt: attempt, # linear backoff
            retry_on_exception=lambda e: isinstance(e, ValueError) # only retry on ValueError
        )

    """

    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            attempt = 1
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt >= max_attempts or not retry_on_exception(e):
                        raise e
                    wait_time = calculate_wait_time(attempt)
                    if retry_message:
                        retry_message(e, attempt, wait_time, *args, **kwargs)
                    sleep(wait_time)
                    attempt += 1

        return wrapper

    return decorator