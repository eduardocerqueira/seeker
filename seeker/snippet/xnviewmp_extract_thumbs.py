#date: 2025-12-18T17:17:08Z
#url: https://api.github.com/gists/96e19e5378ca51e508164f0cb4246ea5
#owner: https://api.github.com/users/IntendedConsequence

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "duckdb>=1.4.3",
# ]
# ///

import functools
import math
import pathlib
import zipfile
import duckdb

from typing import Generic, Iterable, Iterator, TypeVar
import time
import sys
import shutil

T = TypeVar("T")

# tqdm courtesy of tinygrad
class tqdm(Generic[T]):
    def __init__(self, iterable:Iterable[T]|None=None, desc:str='', disable:bool=False,
                 unit:str='it', unit_scale=False, total:int|None=None, rate:int=100):
        self.iterable, self.disable, self.unit, self.unit_scale, self.rate = iterable, disable, unit, unit_scale, rate
        self.st, self.i, self.n, self.skip, self.t = time.perf_counter(), -1, 0, 1, getattr(iterable, "__len__", lambda:0)() if total is None else total
        self.set_description(desc)
        self.update(0)
    def __iter__(self) -> Iterator[T]:
        assert self.iterable is not None, "need an iterable to iterate"
        for item in self.iterable:
            yield item
            self.update(1)
        self.update(close=True)
    def __enter__(self): return self
    def __exit__(self, *_): self.update(close=True)
    def set_description(self, desc:str): self.desc = f"{desc}: " if desc else ""
    def update(self, n:int=0, close:bool=False):
        self.n, self.i = self.n+n, self.i+1
        if self.disable or (not close and self.i % self.skip != 0): return
        prog, elapsed, ncols = self.n/self.t if self.t else 0, time.perf_counter()-self.st, shutil.get_terminal_size().columns
        if elapsed and self.i/elapsed > self.rate and self.i: self.skip = max(int(self.i/elapsed)//self.rate,1)
        def HMS(t): return ':'.join(f'{x:02d}' if i else str(x) for i,x in enumerate([int(t)//3600,int(t)%3600//60,int(t)%60]) if i or x)
        def SI(x):
            return (f"{x/1000**int(g:=round(math.log(x,1000),6)):.{int(3-3*math.fmod(g,1))}f}"[:4].rstrip('.')+' kMGTPEZY'[int(g)].strip()) if x else '0.00'
        prog_text = f'{SI(self.n)}{f"/{SI(self.t)}" if self.t else self.unit}' if self.unit_scale else f'{self.n}{f"/{self.t}" if self.t else self.unit}'
        est_text = f'<{HMS(elapsed/prog-elapsed) if self.n else "?"}' if self.t else ''
        it_text = (SI(self.n/elapsed) if self.unit_scale else f"{self.n/elapsed:5.2f}") if self.n else "?"
        suf = f'{prog_text} [{HMS(elapsed)}{est_text}, {it_text}{self.unit}/s]'
        sz = max(ncols-len(self.desc)-3-2-2-len(suf), 1)
        bar = '\r' + self.desc + (f'{100*prog:3.0f}%|{("█"*int(num:=sz*prog)+" ▏▎▍▌▋▊▉"[int(8*num)%8].strip()).ljust(sz," ")}| ' if self.t else '') + suf
        print(bar[:ncols+1], flush=True, end='\n'*close, file=sys.stderr)
    @classmethod
    def write(cls, s:str): print(f"\r\033[K{s}", flush=True, file=sys.stderr)

class trange(tqdm):
    def __init__(self, n:int, **kwargs): super().__init__(iterable=range(n), total=n, **kwargs)


@functools.cache
def gettable(offset):
    table = bytes((i + offset) & 0xFF for i in range(256))
    return table


def deob(data):
    offset = data[0]
    b = data[0x25:]
    b = b.replace(b'\x01\x01', b'\x00')
    b = b.replace(b'\x01\x03', b'\x27')
    b = b.replace(b'\x01\x02', b'\x01')

    table = gettable(offset)
    return b.translate(table)


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Extract and deobfuscate JPEG thumbnails from XnViewMP Thumbs.db SQLite database."
    )

    parser.add_argument(
        "db",
        type=pathlib.Path,
        help="Path to thumbnail database"
    )

    parser.add_argument(
        "-o", "--out",
        type=pathlib.Path,
        help="Output zip path (default: <db>.zip)"
    )

    parser.add_argument(
        "-f", "--force",
        action="store_true",
        help="Overwrite output file if it exists"
    )

    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress output"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write output, just report what would be done"
    )

    args = parser.parse_args()

    db_path = args.db
    out_path = args.out or db_path.with_suffix(".zip")

    if out_path.exists() and not args.force:
        parser.error(f"{out_path} exists (use --force to overwrite)")

    db = duckdb.connect(db_path, read_only=True)

    query_count = "SELECT count(*) FROM Datas WHERE data IS NOT NULL"
    count, = db.query(query_count).fetchone()

    if args.dry_run:
        print(f"{count} thumbnails found")
        return

    cur = db.execute(
        "SELECT imageid, data FROM Datas WHERE data IS NOT NULL"
    )

    iterator = None if args.quiet else tqdm(None, total=count)

    BATCH_SIZE = 1024

    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_STORED) as f:
        while True:
            rows = cur.fetchmany(BATCH_SIZE)
            if not rows:
                break

            for imageid, data in rows:
                try:
                    thumb = deob(data)
                    f.writestr(f"{imageid}.jpg", thumb)
                except Exception as e:
                    print(e, file=sys.stderr)
            if not args.quiet:
                iterator.update(len(rows))
    if not args.quiet:
        iterator.update(0, close=True)
    db.close()

if __name__ == "__main__":
    main()