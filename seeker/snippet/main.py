#date: 2021-12-01T17:09:04Z
#url: https://api.github.com/gists/f6f7e5ddb30f56291db3b0c67945da64
#owner: https://api.github.com/users/mypy-play

from typing import Dict, List, Any
import select
import sys

class _epoll():
		""" #!if windows
		Create a epoll() implementation that simulates the epoll() behavior.
		This so that the rest of the code doesn't need to worry weither we're using select() or epoll().
		"""
		def __init__(self) -> None:
			self.sockets: Dict[str, Any] = {}
			self.monitoring: Dict[int, Any] = {}

		def unregister(self, fileno :int, *args :List[Any], **kwargs :Dict[str, Any]) -> None:
			try:
				del(self.monitoring[fileno])
			except:
				pass

		def register(self, fileno :int, *args :List[Any], **kwargs :Dict[str, Any]) -> None:
			self.monitoring[fileno] = True

		def poll(self, timeout: float = 0.05, *args :List[Any], **kwargs :Dict[str, Any]) -> List[Any]:
			try:
				return [[fileno, 1] for fileno in select.select(list(self.monitoring.keys()), [], [], timeout)[0]]
			except OSError:
				return []


if sys.platform != "win32":
	from select import epoll, EPOLLIN, EPOLLHUP
else:
	EPOLLIN: Final = 0
	EPOLLHUP: Final = 0
	epoll = _epoll
	select.__dict__['epoll'] = _epoll