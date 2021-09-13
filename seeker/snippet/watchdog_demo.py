#date: 2021-09-13T17:00:16Z
#url: https://api.github.com/gists/0c1585cfe595888f7ba049f9859073ec
#owner: https://api.github.com/users/jupiterbjy

import time
import traceback

import h5py
import queue
from typing import Union

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, DirCreatedEvent, FileCreatedEvent

from loguru import logger


class NewFileHandler(FileSystemEventHandler):
    """h5 file creation handler for Watchdog"""

    def __init__(self):
        self.file_queue = queue.Queue()

    # callback for File/Directory created event, called by Observer.
    def on_created(self, event: Union[DirCreatedEvent, FileCreatedEvent]):

        if event.src_path[-3:] == ".h5":

            logger.debug("New h5 file at \"{}\"", event.src_path)

            # run callback with path string
            self.file_queue.put(event.src_path)


class ObserverWrapper:
    """Encapsulated Observer boilerplate"""

    def __init__(self, path: str, recursive=True):
        self.path = path
        self.recursive = recursive

        self.observer = Observer()
        self.handler = NewFileHandler()

        self.observer.schedule(self.handler, path=path, recursive=recursive)

        self.start()

    def start(self):
        """
        Starts observing for filesystem events. Runs self.routine() every 1 second.

        :param blocking: If true, blocks main thread until keyboard interrupt.
        """

        self.observer.start()
        logger.debug("Observer {} started", id(self))
        logger.debug("Path: \"{}\", Recursive: {}", self.path, self.recursive)

    def stop(self):
        """
        Stops the observer. When running self.start(blocking=True) then you don't need to call this.
        """

        self.observer.stop()
        self.observer.join()

    def wait_for_file(self):
        """
        Wait and Process newly created files
        """

        max_retry_count = 5
        retry_interval_seconds = 6

        # wait for file to be added
        file_path = self.handler.file_queue.get(block=True)

        logger.debug("Got a file {}", file_path)

        # try to open the file
        retry_count = 0
        try:
            file = h5py.File(file_path, "r")
        except OSError:
            if retry_count < max_retry_count:
                retry_count += 1
                print(f"h5 file <{file_path}> is locked, retrying {retry_count}/{max_retry_count}")
                time.sleep(retry_interval_seconds)
            else:
                print(f"h5 file <{file_path}> reached max retry count, skipping")
                return None
        except Exception as err:
            print(f"Got unexpected Error <{type(err).__name__}> while opening <{file_path}> ")
            traceback.print_exc()
        else:
            file.close()

            return file_path

