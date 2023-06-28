#date: 2023-06-28T16:50:14Z
#url: https://api.github.com/gists/b5aedecf09ab51ed614cf5259cb80b13
#owner: https://api.github.com/users/elicharlese

# USING DEQUE
class RecentCounter:

    def __init__(self):
        self.queue = deque()

    def ping(self, t: int) -> int:
        queue = self.queue
        start = t - 3000
        queue.append(t)
        while(queue and queue[0] < start):
            queue.popleft()
        return len(queue)