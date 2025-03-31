#date: 2025-03-31T16:53:49Z
#url: https://api.github.com/gists/b091adf8b1e58e9ebeab044d131019b9
#owner: https://api.github.com/users/amulyavarshney

import unittest
import time
import logging
from io import StringIO
from contextlib import redirect_stdout

# Import the rate limiter implementations
from rate_limiter import (
    TokenBucketRateLimiter, 
    FixedWindowRateLimiter, 
    RateLimiterService
)

class TestRateLimiterE2E(unittest.TestCase):
    """End-to-end tests for rate limiter implementation."""
    
    def setUp(self):
        """Set up test environment."""
        # Configure logging to a string buffer for testing
        self.log_stream = StringIO()
        self.handler = logging.StreamHandler(self.log_stream)
        self.logger = logging.getLogger("test_rate_limiter")
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(self.handler)
        
        # Clear any previous handlers to avoid duplicate logs
        self.logger.handlers = [self.handler]
    
 "**********"  "**********"  "**********"  "**********"  "**********"d "**********"e "**********"f "**********"  "**********"t "**********"e "**********"s "**********"t "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"_ "**********"b "**********"u "**********"c "**********"k "**********"e "**********"t "**********"_ "**********"b "**********"a "**********"s "**********"i "**********"c "**********"_ "**********"f "**********"u "**********"n "**********"c "**********"t "**********"i "**********"o "**********"n "**********"a "**********"l "**********"i "**********"t "**********"y "**********"( "**********"s "**********"e "**********"l "**********"f "**********") "**********": "**********"
        """Test basic functionality of token bucket rate limiter."""
        # Create rate limiter with 3 requests per second
        limiter = "**********"=3, bucket_capacity=3)
        service = RateLimiterService(limiter, self.logger)
        
        # Process 5 requests in quick succession
        results = []
        for i in range(5):
            result = service.process_request("test_user", {"amount": 100})
            results.append(result["status"])
        
        # First 3 should succeed, next 2 should fail
        self.assertEqual(results, ["success", "success", "success", "error", "error"])
        
        # Wait for bucket to refill
        time.sleep(1.1)
        
        # Next request should succeed
        result = service.process_request("test_user", {"amount": 100})
        self.assertEqual(result["status"], "success")
    
    def test_fixed_window_basic_functionality(self):
        """Test basic functionality of fixed window rate limiter."""
        # Create rate limiter with 3 requests per second
        limiter = FixedWindowRateLimiter(rate_limit=3, window_size=1)
        service = RateLimiterService(limiter, self.logger)
        
        # Process 5 requests in quick succession
        results = []
        for i in range(5):
            result = service.process_request("test_user", {"amount": 100})
            results.append(result["status"])
        
        # First 3 should succeed, next 2 should fail
        self.assertEqual(results, ["success", "success", "success", "error", "error"])
        
        # Wait for window to reset
        time.sleep(1.1)
        
        # Next request should succeed
        result = service.process_request("test_user", {"amount": 100})
        self.assertEqual(result["status"], "success")
    
    def test_multiple_users(self):
        """Test that rate limiting is applied per user."""
        limiter = "**********"=2, bucket_capacity=2)
        service = RateLimiterService(limiter, self.logger)
        
        # User 1 makes 3 requests
        user1_results = []
        for i in range(3):
            result = service.process_request("user1", {"amount": 100})
            user1_results.append(result["status"])
        
        # User 2 makes 3 requests
        user2_results = []
        for i in range(3):
            result = service.process_request("user2", {"amount": 100})
            user2_results.append(result["status"])
        
        # Each user should get 2 successes and 1 failure
        self.assertEqual(user1_results, ["success", "success", "error"])
        self.assertEqual(user2_results, ["success", "success", "error"])
    
    def test_burst_handling(self):
        """Test how token bucket handles bursts of traffic."""
        # Create limiter with 2 req/sec rate and 4 burst capacity
        limiter = "**********"=2, bucket_capacity=4)
        service = RateLimiterService(limiter, self.logger)
        
        # User should be able to make 4 requests immediately (burst capacity)
        results = []
        for i in range(6):
            result = service.process_request("burst_user", {"amount": 100})
            results.append(result["status"])
        
        # First 4 should succeed, next 2 should fail
        self.assertEqual(results, ["success", "success", "success", "success", "error", "error"])
        
        # Wait for partial refill (1 second = "**********"
        time.sleep(1.1)
        
        # Should now be able to make 2 more requests
        results = []
        for i in range(3):
            result = service.process_request("burst_user", {"amount": 100})
            results.append(result["status"])
        
        # First 2 should succeed, 3rd should fail
        self.assertEqual(results, ["success", "success", "error"])
    
    def test_window_boundary(self):
        """Test behavior at fixed window boundaries."""
        limiter = FixedWindowRateLimiter(rate_limit=3, window_size=1)
        service = RateLimiterService(limiter, self.logger)
        
        # Use up the limit
        for i in range(3):
            service.process_request("window_user", {"amount": 100})
        
        # 4th request should fail
        result = service.process_request("window_user", {"amount": 100})
        self.assertEqual(result["status"], "error")
        
        # Wait just over the window size
        time.sleep(1.01)
        
        # Should be able to make 3 more requests
        results = []
        for i in range(4):
            result = service.process_request("window_user", {"amount": 100})
            results.append(result["status"])
        
        # First 3 should succeed, 4th should fail
        self.assertEqual(results, ["success", "success", "success", "error"])
    
    def test_error_handling(self):
        """Test error handling in the rate limiter service."""
        # Create a faulty rate limiter that raises exceptions
        class FaultyRateLimiter:
            def allow_request(self, user_id):
                raise Exception("Simulated failure")
        
        faulty_limiter = FaultyRateLimiter()
        service = RateLimiterService(faulty_limiter, self.logger)
        
        # Process should handle the exception gracefully
        result = service.process_request("error_user", {"amount": 100})
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["message"], "Internal service error")
        
        # Check that error was logged
        log_content = self.log_stream.getvalue()
        self.assertIn("Error processing request for user error_user", log_content)
        self.assertIn("Simulated failure", log_content)

if __name__ == "__main__":
    unittest.main()