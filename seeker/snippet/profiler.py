#date: 2025-12-02T16:58:19Z
#url: https://api.github.com/gists/9c0942fe1b1b4a26d20954f84b38be44
#owner: https://api.github.com/users/Fr4nk1inCs

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Self, cast

import torch
from torch.cuda import _POOL_HANDLE, Event


class Timer:
    def __init__(self):
        self._start_event: Event = Event(enable_timing=True, external=True)
        self._end_event: Event = Event(enable_timing=True, external=True)

    @contextmanager
    def __call__(self):
        self._start_event.record()
        yield
        self._end_event.record()

    def elapsed_time(self) -> float:
        return self._start_event.elapsed_time(self._end_event)

    def synchronize(self):
        self._start_event.synchronize()
        self._end_event.synchronize()


@dataclass
class _TimingRecord[Context]:
    context: Context
    timer: Timer


class TimerRegistry[Context]:
    def __init__(self):
        self._timings: list[_TimingRecord[Context]] = []

    @contextmanager
    def time(self, context: Context):
        timer = Timer()
        timing = _TimingRecord(context=context, timer=timer)
        self._timings.append(timing)

        with timer():
            yield

    def elapsed_times(self) -> Iterator[tuple[Context, float]]:
        for timing in self._timings:
            timing.timer.synchronize()
            yield timing.context, timing.timer.elapsed_time()


class Profiler[Context]:
    singleton: Any = None

    def __init__(self, dump_fn: Callable[[Context, float], None]):
        self._graph2registry: dict[torch.cuda.CUDAGraph, TimerRegistry[Context]] = {}
        self._dump_fn = dump_fn
        self._current_graph: torch.cuda.CUDAGraph | None = None
        self._hook_torch_cuda_graph()

    def __new__(cls, dump_fn: Callable[[Context, float], None]) -> Self:
        if cls.singleton is None:
            cls.singleton = super(Profiler, cls).__new__(cls)
        return cast(Self, cls.singleton)

    def _hook_torch_cuda_graph(self):
        """Hook into CUDA graph APIs to manage TimerRegistries automatically."""

        profiler = self
        original_cg_init = torch.cuda.CUDAGraph.__init__
        original_cg_enter = torch.cuda.CUDAGraph.capture_begin
        original_cg_exit = torch.cuda.CUDAGraph.capture_end
        original_cg_replay = torch.cuda.CUDAGraph.replay

        def hooked_cg_init(self: torch.cuda.CUDAGraph):
            original_cg_init(self)
            profiler._graph2registry[self] = TimerRegistry()

        def hooked_cg_enter(
            self: torch.cuda.CUDAGraph,
            pool: _POOL_HANDLE | None = None,
            capture_error_mode: str = "global",
        ) -> None:
            profiler._current_graph = self
            profiler._graph2registry[self]._timings.clear()
            original_cg_enter(self, pool, capture_error_mode)

        def hooked_cg_exit(self: torch.cuda.CUDAGraph) -> None:
            original_cg_exit(self)
            profiler._current_graph = None

        def hooked_cg_replay(self: torch.cuda.CUDAGraph):
            original_cg_replay(self)
            registry = profiler._graph2registry.get(self, None)
            if registry is not None:
                for context, elapsed in registry.elapsed_times():
                    profiler._dump_fn(context, elapsed)

        torch.cuda.CUDAGraph.__init__ = hooked_cg_init
        torch.cuda.CUDAGraph.capture_begin = hooked_cg_enter
        torch.cuda.CUDAGraph.capture_end = hooked_cg_exit
        torch.cuda.CUDAGraph.replay = hooked_cg_replay

    @contextmanager
    def time(self, context: Context):
        """Time a code block within the current CUDA graph context."""
        if self._current_graph is not None:
            registry = self._graph2registry[self._current_graph]
            with registry.time(context=context):
                yield
        else:
            timer = Timer()
            with timer():
                yield
            self._dump_fn(context, timer.elapsed_time())