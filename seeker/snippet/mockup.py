#date: 2025-02-05T16:51:04Z
#url: https://api.github.com/gists/5e107e40034e198aa5d99beadc15c57a
#owner: https://api.github.com/users/callumforrester

import itertools
import time
from pathlib import Path
from typing import Any

import bluesky.plan_stubs as bps
import bluesky.plans as bp
import bluesky.preprocessors as bpp
from bluesky.protocols import (
    DataKey,
    Flyable,
    Preparable,
    Readable,
    Reading,
    Stageable,
    Triggerable,
)
from bluesky.run_engine import RunEngine
from bluesky.utils import Msg, MsgGenerator
from event_model import RunStart
from ophyd_async.core import AsyncStatus, PathInfo, PathProvider

# Fake ophyd-async


class Detector(Readable, Triggerable, Stageable, Preparable, Flyable):
    def __init__(self, name: str, provider: PathProvider) -> None:
        super().__init__()
        self._name = name
        self._provider = provider
        self.parent = None

    # CHANGE_REQURED
    # Would need to add to the PathProvider API so that detectors can
    # register/deregister themselves
    @AsyncStatus.wrap
    async def stage(self):
        self._provider.on_stage(self.name)

    @AsyncStatus.wrap
    async def unstage(self):
        self._provider.on_unstage(self.name)

    @AsyncStatus.wrap
    async def prepare(self, value) -> None:
        inf = self._provider(self.name)
        print(f"{self.name} --> {inf}")

    @property
    def name(self) -> str:
        return self._name

    @AsyncStatus.wrap
    async def trigger(self) -> None:
        await self.prepare(None)

    async def describe(self) -> dict[str, DataKey]:
        return {self.name: DataKey(dtype="number", shape=[1], source="foo")}

    async def read(self) -> dict[str, Reading]:
        return {self.name: Reading(value=0.0, timestamp=time.time())}

    @AsyncStatus.wrap
    async def kickoff(self) -> None: ...

    @AsyncStatus.wrap
    async def complete(self) -> None: ...


# Fake numtracker

scan_number = itertools.count()


async def fetch_from_numtracker(visit: str, detectors: list[str]) -> dict[str, Any]:
    num = next(scan_number)
    return {
        "scanNumber": num,
        "scanFile": f"/data/{visit}/{num}",
        "detectors": {
            detector: Path(f"/data/{visit}/{detector}-{num}") for detector in detectors
        },
    }


# blueapi.provider


class CachedPathProvider(PathProvider):
    def __init__(self) -> None:
        super().__init__()
        self._staged = []
        self._cache: dict[str, PathInfo] | None = None

    def __call__(self, device_name: str | None = None) -> PathInfo:
        if self._cache is None:
            raise KeyError("No cache")
        elif device_name is None:
            raise KeyError("Must provide a device name")
        else:
            return self._cache[device_name]

    # CHANGE REQURIED
    # Additions to API from blueapi's point of view
    def update(self, paths: dict[str, PathInfo]) -> None:
        self._cache = paths

    # CHANGE REQURIED
    # Additions to API from detector's point of view
    def on_stage(self, device_name: str) -> None:
        self._staged.append(device_name)

    def on_unstage(self, device_name: str) -> None:
        self._staged.remove(device_name)

    @property
    def staged(self) -> list[str]:
        return self._staged


# At this point the API of PathProvider is doing a lot more than providing a path,
# it is being updated in three different ways, this seems like poor design smell


provider = CachedPathProvider()


# blueapi.context


async def update_from_numtracker(md) -> dict[str, Any]:
    visit = md["visit"]
    detectors = provider.staged
    data_collection = await fetch_from_numtracker(visit, detectors)
    provider.update(
        {
            detector: PathInfo(directory_path=path, filename=path.name + ".h5")
            for detector, path in data_collection["detectors"].items()
        }
    )
    md["scan_file"] = data_collection["scanFile"]
    return data_collection["scanNumber"]


# CHANGE REQUIRED
# The RE currently only accepts a sync function for scan_id_source, making it
# accept either a sync or async function is a trivial change and backwards
# compatible, but would need to get it approved etc.
RE = RunEngine(scan_id_source=update_from_numtracker)


d1 = Detector("d1", provider=provider)
d2 = Detector("d2", provider=provider)
d3 = Detector("d3", provider=provider)


# blueapi.worker


def one_run_one_file() -> MsgGenerator[None]:
    yield from bp.count([d1, d2])


def different_runs_different_files() -> MsgGenerator[None]:
    yield from bp.count([d1])
    yield from bp.count([d2, d3])
    yield from bp.count([d1, d2, d3])


# This takes use case #3 in https://github.com/bluesky/bluesky/issues/1849 very
# literally if we do this we can never prepare before the first run start,
# which leads to additional code complexity here
def different_runs_one_file() -> MsgGenerator[None]:
    @bpp.run_decorator()
    def optimized_scan(detectors: list[Detector], prepare: bool = False):
        if prepare:
            for d in all_detectors:
                yield from bps.prepare(d, None)

        for d in detectors:
            yield from bps.kickoff(d)
        for d in detectors:
            yield from bps.complete(d)

    all_detectors = [d1, d2, d3]

    @bpp.stage_decorator(all_detectors)
    def _inner_plan() -> MsgGenerator[None]:
        yield from optimized_scan([d1, d2, d3], prepare=True)
        yield from optimized_scan([d1])
        yield from optimized_scan([d2, d3])

    yield from _inner_plan()


def summarize(_: str, start_doc: RunStart):
    from pprint import pprint

    print("New run opened:")
    pprint({key: start_doc.get(key) for key in ["uid", "scan_id", "scan_file"]})


def run_plan(plan: MsgGenerator[Any], visit: str):
    print("=" * 32)
    print(f"Running plan {plan.__name__}")
    RE.md["visit"] = visit
    runs = RE(plan, {"start": summarize})
    print(f"Plan complete: runs: {runs}")
    print()


run_plan(one_run_one_file(), "cm-123456")
run_plan(different_runs_different_files(), "cm-123456")
run_plan(different_runs_one_file(), "cm-123456")
