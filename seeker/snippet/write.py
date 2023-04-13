#date: 2023-04-13T17:03:52Z
#url: https://api.github.com/gists/12e7e5c4417049af22fbe7fcf8f01134
#owner: https://api.github.com/users/naderidev

import m3u8
from redis import Redis
from rq import Queue

class Writer:

    def __init__(self, writer: str, **kwargs) -> None:
        super().__init__()
        self.queue = Queue(connection=Redis())
        self.writer_name = writer
        self.kwargs = kwargs

    def write(self, segment: m3u8.Segment):
        match self.writer_name:
            case 'local_writer':
                self.queue.enqueue(
                    "writers.local_write",
                    file_path=self.kwargs['file_path'],
                    mode=self.kwargs['mode'],
                    m3u8_base_url=self.kwargs['m3u8_base_url'],
                    segment=segment,
                )

    def stop(self):
        self.queue.empty()
    
    