#date: 2026-02-20T17:25:48Z
#url: https://api.github.com/gists/019fe661e301476638c531c4e6e589d7
#owner: https://api.github.com/users/ramilbakhshyiev

import time
from collections import deque
from threading import Lock

class LastMinuteCounter:
    def __init__(self):
        self.q = deque() # 2-element int arrays
        self.window_sum = 0
        self.lock = Lock()

    def _current_sec(self) -> int:
        return int(time.time())

    def _cleanup(self, now_sec: int) -> None:
        cutoff = now_sec - 60
        while self.q and self.q[0][0] <= cutoff:
            _, old_cnt = self.q.popleft()
            self.window_sum -= old_cnt

    def increment(self, n: int = 1) -> None:
        now = self._current_sec()
        with self.lock:
            self._cleanup(now)

            if self.q and self.q[-1][0] == now:
                self.q[-1][1] += n
            else:
                self.q.append([now, n])
                
            self.window_sum += n

    def get_last_minute(self) -> int:
        now = self._current_sec()
        with self.lock:
            self._cleanup(now)
            return self.window_sum