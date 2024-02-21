#date: 2024-02-21T17:02:03Z
#url: https://api.github.com/gists/59ca96a4377d61a84e653865bc9a3d61
#owner: https://api.github.com/users/mvandermeulen

import redis
import time
import uuid

class RedisMutex:
    def __init__(self, key, host='localhost', port=6379, db=0):
        self.key = key
        self.redis = redis.Redis(host=host, port=port, db=db)
        self.lock_id = uuid.uuid4().hex

    def acquire(self, timeout=10):
        """ Acquire the lock """
        end = time.time() + timeout
        while time.time() < end:
            if self.redis.setnx(self.key, self.lock_id):
                # Lock acquired
                return True
            time.sleep(0.01)  # Wait a little before trying again
        return False

    def release(self):
        """ Release the lock """
        if self.redis.get(self.key) == self.lock_id.encode():
            self.redis.delete(self.key)

    def __enter__(self):
        """ Support for with statement """
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """ Support for with statement """
        self.release()
