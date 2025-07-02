#date: 2025-07-02T17:14:05Z
#url: https://api.github.com/gists/4d9b87dc2dbcc36ebcf1a25b75df67ed
#owner: https://api.github.com/users/sovietscout

import csv
import asyncio
import logging
from pathlib import Path
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor

log = logging.getLogger("FileStream")

class AsyncFileStreamer:
    def __init__(self, filepath: str | Path, fieldnames: List[str], max_rows_per_file: int = 50000):
        self.file = Path(filepath)
        self.file.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = fieldnames
        self.max_rows_per_file = max_rows_per_file
        self.row_count = 0
        self.current_file_row_count = 0
        self.file_counter = 0
        self.base_file = self.file

        self._queue = asyncio.Queue()
        self._writer_task: Optional[asyncio.Task] = None
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._file_handle = None
        self._writer = None

    async def __aenter__(self):
        self.file_counter = 0
        self.current_file = self._get_current_filename()
        self.current_file.parent.mkdir(parents=True, exist_ok=True)
        
        loop = asyncio.get_running_loop()
        self._file_handle = await loop.run_in_executor(
            self._executor,
            lambda: open(self.current_file, "w", newline="", encoding="utf-8")
        )

        self._writer = csv.DictWriter(self._file_handle, fieldnames=self.fieldnames)
        await loop.run_in_executor(self._executor, self._writer.writeheader)
        self._writer_task = asyncio.create_task(self._writer_loop())

        log.debug("File stream opened: %s", self.current_file)
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self._queue.join()

        if self._writer_task:
            self._writer_task.cancel()
            try:
                await self._writer_task
            except asyncio.CancelledError:
                pass

        if self._file_handle:
            await self._close_file()

        self._executor.shutdown(wait=True)
        log.debug("File stream closed. Total files: %d", self.file_counter + 1)

    async def write(self, data: List[dict]):
        self.row_count += len(data)
        await self._queue.put(data)

    async def _write_batch(self, data: List[dict]):
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, lambda: self._writer.writerows(data))

    async def _close_file(self):
        if self._file_handle:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self._executor, self._file_handle.close)
            self._file_handle = None
            self._writer = None

    async def _rotate_file(self):
        await self._close_file()
        self.file_counter += 1
        self.current_file = self._get_current_filename()
        self.current_file.parent.mkdir(parents=True, exist_ok=True)
        self.current_file_row_count = 0

        loop = asyncio.get_running_loop()
        self._file_handle = await loop.run_in_executor(
            self._executor,
            lambda: open(self.current_file, "w", newline="", encoding="utf-8")
        )
        self._writer = csv.DictWriter(self._file_handle, fieldnames=self.fieldnames)
        await loop.run_in_executor(self._executor, self._writer.writeheader)
        log.debug("Rotated to new file: %s", self.current_file)

    def _get_current_filename(self):
        if self.file_counter == 0:
            return self.base_file
        stem = self.base_file.stem
        suffix = self.base_file.suffix
        return self.base_file.with_name(f"{stem}_{self.file_counter}{suffix}")

    async def _writer_loop(self):
        while True:
            data = await self._queue.get()
            if not data:
                self._queue.task_done()
                continue

            try:
                while data:
                    remaining = self.max_rows_per_file - self.current_file_row_count
                    if remaining <= 0:
                        await self._rotate_file()
                        remaining = self.max_rows_per_file

                    chunk = data[:remaining]
                    data = data[remaining:]
                    await self._write_batch(chunk)
                    self.current_file_row_count += len(chunk)
            except Exception as e:
                log.error("Write error: %s", e)
            finally:
                self._queue.task_done()

async def main():
    async with AsyncFileStream(filename="...csv", fieldnames=[...]) as stream:
        data = {"name": "...", ...}
        await stream.write(data)

if __name__ == "__main__":
    asyncio.run(main())
    