#date: 2025-03-31T16:53:49Z
#url: https://api.github.com/gists/b091adf8b1e58e9ebeab044d131019b9
#owner: https://api.github.com/users/amulyavarshney

import time
from collections import defaultdict
from threading import Lock

 "**********"c "**********"l "**********"a "**********"s "**********"s "**********"  "**********"T "**********"o "**********"k "**********"e "**********"n "**********"B "**********"u "**********"c "**********"k "**********"e "**********"t "**********"R "**********"a "**********"t "**********"e "**********"L "**********"i "**********"m "**********"i "**********"t "**********"e "**********"r "**********": "**********"
    """
    Rate limiter using Token Bucket algorithm.
    Limits each user to a specified number of requests per second.
    """
    
    def __init__(self, rate_limit=5, bucket_capacity=5):
        """
        Initialize the rate limiter.
        
        Args:
            rate_limit: "**********": 5)
            bucket_capacity: "**********": 5)
        """
        self.rate_limit = "**********"
        self.bucket_capacity = bucket_capacity
        self.user_buckets = defaultdict(lambda: "**********": bucket_capacity, "last_refill": time.time()})
        self.lock = Lock()  # For thread safety
    
 "**********"  "**********"  "**********"  "**********"  "**********"d "**********"e "**********"f "**********"  "**********"_ "**********"r "**********"e "**********"f "**********"i "**********"l "**********"l "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"s "**********"( "**********"s "**********"e "**********"l "**********"f "**********", "**********"  "**********"u "**********"s "**********"e "**********"r "**********"_ "**********"i "**********"d "**********") "**********": "**********"
        """Refill tokens based on elapsed time since last refill."""
        bucket = self.user_buckets[user_id]
        now = time.time()
        time_passed = now - bucket["last_refill"]
        new_tokens = "**********"
        
        # Update token count and timestamp
        bucket["tokens"] = "**********"
        bucket["last_refill"] = now
    
    def allow_request(self, user_id):
        """
        Check if a request from the specified user is allowed.
        
        Args:
            user_id: The ID of the user making the request
            
        Returns:
            bool: True if request is allowed, False otherwise
        """
        with self.lock:
            self._refill_tokens(user_id)
            
            # Check if there's at least one token available
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"s "**********"e "**********"l "**********"f "**********". "**********"u "**********"s "**********"e "**********"r "**********"_ "**********"b "**********"u "**********"c "**********"k "**********"e "**********"t "**********"s "**********"[ "**********"u "**********"s "**********"e "**********"r "**********"_ "**********"i "**********"d "**********"] "**********"[ "**********"" "**********"t "**********"o "**********"k "**********"e "**********"n "**********"s "**********"" "**********"] "**********"  "**********"> "**********"= "**********"  "**********"1 "**********": "**********"
                self.user_buckets[user_id]["tokens"] -= "**********"
                return True
            return False


class FixedWindowRateLimiter:
    """
    Rate limiter using Fixed Window Counter algorithm.
    Limits each user to a specified number of requests per time window.
    """
    
    def __init__(self, rate_limit=5, window_size=1):
        """
        Initialize the rate limiter.
        
        Args:
            rate_limit: Maximum number of requests allowed in a window (default: 5)
            window_size: Size of the time window in seconds (default: 1)
        """
        self.rate_limit = rate_limit
        self.window_size = window_size
        self.user_counters = defaultdict(lambda: {"count": 0, "window_start": time.time()})
        self.lock = Lock()  # For thread safety
    
    def allow_request(self, user_id):
        """
        Check if a request from the specified user is allowed.
        
        Args:
            user_id: The ID of the user making the request
            
        Returns:
            bool: True if request is allowed, False otherwise
        """
        with self.lock:
            counter = self.user_counters[user_id]
            current_time = time.time()
            
            # Check if we're in a new time window
            if current_time - counter["window_start"] >= self.window_size:
                # Reset for new window
                counter["count"] = 0
                counter["window_start"] = current_time
            
            # Check if request is within rate limit
            if counter["count"] < self.rate_limit:
                counter["count"] += 1
                return True
            return False


# Error handling and logging wrapper
class RateLimiterService:
    """
    Service that provides rate limiting with error handling and logging.
    """
    
    def __init__(self, rate_limiter, logger=None):
        """
        Initialize the service.
        
        Args:
            rate_limiter: An instance of a rate limiter class
            logger: Logger object (optional)
        """
        self.rate_limiter = rate_limiter
        self.logger = logger
    
    def process_request(self, user_id, transaction_data):
        """
        Process a transaction request with rate limiting.
        
        Args:
            user_id: The ID of the user making the request
            transaction_data: Data related to the transaction
            
        Returns:
            dict: Response containing status and message
        """
        try:
            if self.rate_limiter.allow_request(user_id):
                # Process the transaction (this would call the actual transaction processing logic)
                if self.logger:
                    self.logger.info(f"Transaction processed for user {user_id}")
                return {"status": "success", "message": "Transaction processed"}
            else:
                if self.logger:
                    self.logger.warning(f"Rate limit exceeded for user {user_id}")
                return {"status": "error", "message": "Rate limit exceeded. Please try again later."}
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error processing request for user {user_id}: {str(e)}")
            return {"status": "error", "message": "Internal service error"}


# Example usage:
if __name__ == "__main__":
    import logging
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("rate_limiter_service")
    
    # Create rate limiter service with token bucket algorithm
    token_limiter = "**********"=5, bucket_capacity=5)
    service = "**********"
    
    # Example usage
    for i in range(10):
        result = service.process_request("user123", {"amount": 100, "recipient": "user456"})
        print(f"Request {i+1}: {result}")
        time.sleep(0.1)  # Small delay between requests