#date: 2022-03-23T17:05:26Z
#url: https://api.github.com/gists/12e9770a0fafd79425a56edb12938b39
#owner: https://api.github.com/users/orfins

from threading import Event as _Event, Lock as _Lock


class CountdownEvent:
    """
    Behaves like threading.Event but requires multiple .set() calls
    to let .wait() return
    """
    def __init__(self, count):
        self._count = count
        self._lock = _Lock()
        self._event = _Event()

    def set(self):
        with self._lock:
            self._count -= 1

            if self._count == 0:
                self._event.set()

    def wait(self, timeout=None):
        self._event.wait(timeout)
