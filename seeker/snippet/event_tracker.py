#date: 2024-11-05T16:59:01Z
#url: https://api.github.com/gists/6a8ec968c965132277419589ca10d77e
#owner: https://api.github.com/users/michaeltelford


# Records events for a given key e.g. ShareID in the form of:
#
# {
#   key1: [timestamp1, timestamp2],
#   key2: [timestamp3, timestamp4]
# }
#
# Then tells you when a given threshold/limit has been exceeded.
# Starts a background thread to evict events when their TTL expires.
# 
# Typically, this class is used to simulate 429 logic when the workload increases.
# Can also be used to avoid duplicating logic within a given timeframe (TTL) etc.
#
# Use in a REST endpoint handler for recording HTTP requests like:
#
# requests = EventTracker(1, 10) # 1 event every 10 seconds is the limit
# requests.append("share1")
# if requests.exceeds_threshold("share1"):
#     <return 409 HTTP response etc>
# <return normal HTTP response e.g. 200>
#
# The above snippet will return a 409 if there's more than 1 request within 10 seconds.

import pprint
import threading
import time
from datetime import datetime, timezone, timedelta

def acquire_lock(func):
    def wrapper(*args):
        args[0].mutex.acquire()
        try:
            return func(*args)
        finally:
            if args[0].mutex.locked:
                args[0].mutex.release()

    return wrapper

class EventTracker:
    def __init__(self, event_threshold, event_ttl):
        self.event_threshold = event_threshold
        self.event_ttl = event_ttl
        self.events = {}

        self._start_evict_thread()

    def _start_evict_thread(self):
        self.stop_evicting = False
        self.mutex = threading.Lock()
        self.evict_thread = threading.Thread(target=self._evict)
        self.evict_thread.start()

        pprint.pprint("Event Tracker - Eviction thread started")

    def stop_evict_thread(self):
        self.stop_evicting = True
        self.evict_thread.join()

        pprint.pprint("Event Tracker - Eviction thread stopped")

    @acquire_lock
    def append(self, key):
        if key not in self.events:
            self.events[key] = []

        now = datetime.now(timezone.utc)
        self.events[key].append(now)

        pprint.pprint("Event Tracker - Event with key appended: %s" % (key))

    @acquire_lock
    def exceeds_threshold(self, key):
        if key not in self.events:
            pprint.pprint("Event Tracker - Exceeds threshold result: %s -> %r" % (key, False))
            return False

        timestamps = self.events[key]
        num_events = len(timestamps)
        result = num_events > self.event_threshold

        pprint.pprint("Event Tracker - Exceeds threshold result: %s -> %r" % (key, result))
        return result

    def _evict(self):
        while True:
            if self.stop_evicting:
                break

            time.sleep(1)
            self.mutex.acquire()

            # Remove any expired events (based on their TTL)
            now = datetime.now(timezone.utc)
            for key, timestamps in self.events.items():
                for timestamp in timestamps:
                    deadline = timestamp + timedelta(seconds=self.event_ttl)
                    if now > deadline:
                        timestamps.remove(timestamp)
                        pprint.pprint("Event Tracker - Evicted event for key: %s" % (key))

            # Remove any keys that now have zero events (to lower memory footprint)
            for key, timestamps in self.events.copy().items():
                if len(timestamps) == 0:
                    del self.events[key]

            if self.mutex.locked:
                self.mutex.release()


if __name__ == "__main__":
    event_tracker = EventTracker(1, 3)
    event_tracker.append("share1")
    pprint.pprint(event_tracker.events) # len: 1 events
    pprint.pprint(event_tracker.exceeds_threshold("share1")) # False

    event_tracker.append("share1")
    event_tracker.append("share1")
    pprint.pprint(event_tracker.events) # len: 3 events
    pprint.pprint(event_tracker.exceeds_threshold("share1")) # True
    time.sleep(5)

    pprint.pprint(event_tracker.events) # len: 0 events (with no keys)
    pprint.pprint(event_tracker.exceeds_threshold("share1")) # False
    pprint.pprint(event_tracker.exceeds_threshold("share2")) # False

    event_tracker.stop_evict_thread()
