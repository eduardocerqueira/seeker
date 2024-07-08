#date: 2024-07-08T17:11:13Z
#url: https://api.github.com/gists/8bd5f9bb0263fbf3297de26edd3028f1
#owner: https://api.github.com/users/LogicDaemon

import os
import pathlib
import sys
import timeit
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, Future

base_path = pathlib.Path(os.environ["LOCALAPPDATA"])
file_to_find = "Local State"
n_repeats = int(sys.argv[1])


def find_snaps_rglob():
    paths = []
    for path in base_path.rglob(file_to_find):
        paths.append(path)
    return paths


def find_snaps_walk():
    paths = []
    for dp, dn, fn in os.walk(base_path):
        for f in fn:
            if f == file_to_find:
                paths.append(pathlib.Path(os.path.join(dp, f)))
    return paths


def sighup_futures_process(d: Path) -> list[Path]:
    return list(d.rglob(file_to_find))


def sighup_futures() -> list[Path]:
    top = base_path / file_to_find
    locations: list[Path] = [top] if top.exists() else []
    futures: list[Future] = []
    with ThreadPoolExecutor() as exe:
        for t in base_path.glob("*"):
            if t.is_dir():
                futures.append(exe.submit(sighup_futures_process, t))
    for t in futures:
        locations.extend(t.result())
    return locations


# warmup
assert sorted(find_snaps_walk()) == sorted(sighup_futures())

i = 0
while True:
    i += 1
    rglob_time = timeit.timeit(find_snaps_rglob, number=n_repeats)
    print(f"#{i}: {rglob_time=}")
    walk_time = timeit.timeit(find_snaps_walk, number=n_repeats)
    print(f"#{i}: {walk_time=}")
    sighup_futures_time = timeit.timeit(sighup_futures, number=n_repeats)
    print(f"#{i}: {sighup_futures_time=}")
