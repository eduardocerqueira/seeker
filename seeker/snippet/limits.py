#date: 2026-01-20T17:10:21Z
#url: https://api.github.com/gists/ef2ab7b1d21927bf0a991fe62bc68653
#owner: https://api.github.com/users/jqmviegas

import functools
import time
from typing import Callable, Optional

import httpx
from limits import parse, storage, strategies
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


class RateLimitExceeded(Exception):
    def __init__(self, retry_after: float):
        self.retry_after = retry_after
        super().__init__(f"Rate limit exceeded. Retry after {retry_after:.1f}s")


_limiter: Optional[strategies.MovingWindowRateLimiter] = None


def configure(redis_url: str = "redis://localhost:6379"):
    global _limiter
    backend = storage.RedisStorage(redis_url)
    _limiter = strategies.MovingWindowRateLimiter(backend)


def rate_limit(
    limit: str,
    key: Optional[Callable[..., str]] = None,
    max_retries: int = 3,
    auto_retry: bool = True,
):
    parsed_limit = parse(limit)
    
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if _limiter is None:
                raise RuntimeError("Call configure() first")
            
            identifier = key(*args, **kwargs) if key else func.__name__
            
            for attempt in range(max_retries + 1):
                if _limiter.hit(parsed_limit, func.__module__, identifier):
                    return func(*args, **kwargs)
                
                window = _limiter.get_window_stats(parsed_limit, func.__module__, identifier)
                retry_after = max(0.1, window.reset_time - time.time())
                
                if not auto_retry or attempt == max_retries:
                    raise RateLimitExceeded(retry_after)
                
                time.sleep(retry_after)
        
        return wrapper
    return decorator


# ============ Usage ============

configure("redis://localhost:6379")


# Tenacity on outside: retries network errors
# rate_limit on inside: handles throttling
@retry(
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    stop=stop_after_attempt(3),
)
@rate_limit("10/minute")
def fetch_api(endpoint: str) -> dict:
    """Fetch from external API with rate limiting + retry on network errors."""
    with httpx.Client(timeout=5) as client:
        response = client.get(f"https://api.example.com/{endpoint}")
        response.raise_for_status()
        return response.json()


# Per-user rate limiting with retries
@retry(
    retry=retry_if_exception_type(httpx.HTTPStatusError),
    wait=wait_exponential(multiplier=0.5, max=5),
    stop=stop_after_attempt(3),
)
@rate_limit("5/second", key=lambda user_id: user_id)
def get_user_data(user_id: str) -> dict:
    """Fetch user data with per-user rate limiting."""
    with httpx.Client(timeout=5) as client:
        response = client.get(f"https://api.example.com/users/{user_id}")
        response.raise_for_status()
        return response.json()


if __name__ == "__main__":
    # Simulated usage
    for i in range(3):
        try:
            result = fetch_api("data")
            print(f"Success: {result}")
        except RateLimitExceeded as e:
            print(f"Rate limited: {e}")
        except httpx.HTTPError as e:
            print(f"HTTP error after retries: {e}")
```

**Order matters:**
```
@retry(...)      <-- outer: catches network errors, retries with backoff
@rate_limit(...) <-- inner: ensures we don't exceed API limits
def func():