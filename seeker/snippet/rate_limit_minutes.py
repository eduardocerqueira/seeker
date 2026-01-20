#date: 2026-01-20T17:22:18Z
#url: https://api.github.com/gists/cfea8b8bdd4bbf51fc24f6de4fe4c7b5
#owner: https://api.github.com/users/jqmviegas

import functools
import time
from typing import Callable, Optional

import redis
from tenacity import retry, stop_after_attempt, wait_fixed


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded."""
    def __init__(self, limit: int, retry_after: int):
        self.limit = limit
        self.retry_after = retry_after
        super().__init__(f"Rate limit of {limit}/min exceeded. Retry after {retry_after}s")


class RateLimiter:
    """Rate limiter using Redis minute counters."""
    
    _client: Optional[redis.Redis] = None
    
    @classmethod
    def configure(cls, client: redis.Redis) -> None:
        """Configure the Redis client for all rate limiters."""
        cls._client = client
    
    @classmethod
    def get_client(cls) -> redis.Redis:
        if cls._client is None:
            raise RuntimeError("RateLimiter not configured. Call RateLimiter.configure(client) first.")
        return cls._client


def rate_limit(
    requests_per_minute: int,
    key: Optional[str] = None,
    key_prefix: str = "ratelimit",
):
    """
    Rate limiter decorator using Redis minute counters.
    
    Args:
        requests_per_minute: Maximum requests allowed per minute
        key: Custom key for grouping functions (defaults to func.__name__)
        key_prefix: Prefix for Redis keys
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            client = RateLimiter.get_client()
            
            current_minute = int(time.time() // 60)
            rate_key = key or func.__name__
            redis_key = f"{key_prefix}:{rate_key}:{current_minute}"
            
            current_count = client.incr(redis_key)
            
            if current_count == 1:
                client.expire(redis_key, 60)
            
            if current_count > requests_per_minute:
                seconds_until_next_minute = 60 - (int(time.time()) % 60)
                raise RateLimitExceeded(requests_per_minute, seconds_until_next_minute)
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


# Example usage

@retry(stop=stop_after_attempt(5), wait=wait_fixed(2))
@rate_limit(requests_per_minute=60, key="gsheets")
def read_sheet(sheet_id: str) -> dict:
    print(f"Reading sheet: {sheet_id}")
    return {"data": []}


@retry(stop=stop_after_attempt(5), wait=wait_fixed(2))
@rate_limit(requests_per_minute=60, key="gsheets")
def write_sheet(sheet_id: str, data: list) -> dict:
    print(f"Writing to sheet: {sheet_id}")
    return {"status": "success"}


if __name__ == "__main__":
    # Configure once at startup
    RateLimiter.configure(
        redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
    )
    
    # For production, something like:
    # RateLimiter.configure(
    #     redis.Redis.from_url(os.environ["REDIS_URL"], decode_responses=True)
    # )
    
    read_sheet("abc123")
    write_sheet("abc123", [{"col": "value"}])