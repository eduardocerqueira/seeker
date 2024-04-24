#date: 2024-04-24T16:53:09Z
#url: https://api.github.com/gists/a200f2bf20a32d0c9b1c6bce9dea8981
#owner: https://api.github.com/users/dgunning

class Throttler:
    """
    A simple throttler that limits the number of requests per time window
    """

    def __init__(self,
                 request_rate: RequestRate = SecMaxRequestRate):
        self.request_rate = request_rate
        self.request_timestamps = []
        if self.request_rate.time_window == 1:
            denominator = 'per second'
        else:
            denominator = f'every {self.request_rate.time_window} seconds'

    def get_ticket(self):
        current_time = time.time()

        # Remove timestamps older than the time window
        while self.request_timestamps and self.request_timestamps[0] <= current_time - self.request_rate.time_window:
            self.request_timestamps.pop(0)

        if len(self.request_timestamps) < self.request_rate.max_requests:
            self.request_timestamps.append(current_time)
            return True
        else:
            return False

    def wait_for_ticket(self):
        while not self.get_ticket():
            time.sleep(0.1)  # Wait for a short interval before checking again


def throttle_requests(request_rate=None, requests_per_second=None):
    if requests_per_second is not None:
        request_rate = RequestRate(max_requests=requests_per_second, time_window=1)
    elif request_rate is None:
        raise ValueError("Either request_rate or requests_per_second must be provided")

    throttler = Throttler(request_rate)

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            throttler.wait_for_ticket()
            return func(*args, **kwargs)

        return wrapper

    return decorator


@throttle_requests(requests_per_second=10)
def download_json(data_url: str):
    ...

