#date: 2024-02-14T17:08:14Z
#url: https://api.github.com/gists/5c717ae68ac037e72ae45fd1e9ca1345
#owner: https://api.github.com/users/arthur-tacca

# aioresult variant of StreamResultsNursery
# Original idea by smurfix: https://gist.github.com/smurfix/0130817fa5ba6d3bb4a0f00e4d93cf86
# Fixed non-aioresult version: https://gist.github.com/arthur-tacca/6c676a21d0dcc0582edb50c9c2aa3e3c


from collections import deque
import math

from aioresult import ResultCapture
import trio


class StreamResultsNursery:
    """Nursery that streams results to an async for loop.

    This is like the aioresult function to_stream(), except that it allows for further tasks to be
    added. The loop over results until all tasks have finished.

    The async for loop will always return a result for every started task, even if the nursery is
    cancelled. If that happens, the result will show as having ended with exception type Cancelled,
    or not being done (if the routine did not start at all, which happens if the nursery was already
    cancelled before .start_*() was called or if it happened while the task was pending due to
    max_running_tasks). This does not include if .start_*() raises an exception itself (due to the
    nursery having not yet started or already finished, or due to the loop having already finished).

    **Loop completion:**

    The logic that decides when the loop should complete will handle the most common situations.
    The usual one is that you kick off one or more tasks within the body of this nursery, and then
    run the async for loop in this nursery, and new tasks are then run from other tasks or from
    within the loop. That is::

        async with StreamResultsNursery() as srn:
            srn.start_soon(foo, srn)  # foo() can then run other tasks, which can run still more...
            async for r in srn:
                print(r)
                # ... it is also possible to run more tasks from in here ...

    In more subtle situations, it is up to you to ensure that the nursery and async for loop do not
    complete before you are done. For example, it is OK to loop over results in an outer nursery::

        async with trio.open_nursery() as outer_nursery:
            async with StreamResultsNursery() as task_nursery:
                task_nursery.start_soon(start_some_tasks, task_nursery)
                outer_nursery.start_soon(loop_over_results, task_nursery)

    Here, the loop_over_results() function receives the task_nursery as an argument so that it can
    perform the loop. But it is NOT OK (at least, not necessarily OK) for loop_over_results() to
    spawn new tasks, because if there are no open tasks when it does this then the task_nursery
    might already have completed.
    """
    def __init__(self, max_running_tasks=math.inf):
        self._nursery = trio.open_nursery()
        self._results = deque()
        self._unfinished_tasks_count = 0  # Includes both running and waiting to run
        self._capacity_limiter = trio.CapacityLimiter(max_running_tasks)
        self._nm = None
        self._parking_lot = trio.lowlevel.ParkingLot()
        self._loop_finished = False

    @property
    def cancel_scope(self):
        return self._nm.cancel_scope

    @property
    def max_running_tasks(self):
        return self._capacity_limiter.total_tokens

    @max_running_tasks.setter
    def max_running_tasks(self, value):
        self._capacity_limiter.total_tokens = "**********"

    @property
    def running_tasks_count(self):
        return self._capacity_limiter.borrowed_tokens

    async def __aenter__(self):
        self._nm = await self._nursery.__aenter__()
        return self

    def __aexit__(self, *exc):
        return self._nursery.__aexit__(*exc)

    async def _wrap(self, rc: ResultCapture, task_status=trio.TASK_STATUS_IGNORED):
        try:
            async with self._capacity_limiter:
                task_status.started()
                await rc.run()
        finally:
            self._results.append(rc)
            self._unfinished_tasks_count -= 1
            self._parking_lot.unpark()

    def start_soon_rc(self, rc: ResultCapture):
        if self._nm is None:
            raise RuntimeError("Enter context manager before starting tasks")
        if self._loop_finished:
            raise RuntimeError("Loop over results has already completed")
        self._unfinished_tasks_count += 1
        self._nm.start_soon(self._wrap, rc)

    def start_soon(self, p, *a, suppress_exception=False):
        rc = ResultCapture(p, *a, suppress_exception=suppress_exception)
        self.start_soon_rc(rc)
        return rc

    async def start_rc(self, rc: ResultCapture):
        if self._nm is None:
            raise RuntimeError("Enter context manager before starting tasks")
        if self._loop_finished:
            raise RuntimeError("Loop over results has already completed")
        self._unfinished_tasks_count += 1
        await self._nm.start(self._wrap, rc)

    async def start(self, p, *a, suppress_exception=False):
        rc = ResultCapture(p, *a, suppress_exception=suppress_exception)
        await self.start_soon_rc(rc)
        return rc

    def __aiter__(self):
        return self

    async def __anext__(self):
        await trio.lowlevel.checkpoint()  # Ensure this function is always a checkpoint

        while len(self._results) == 0 and self._unfinished_tasks_count != 0:
            await self._parking_lot.park()  # Need to wait for a result to be produced

        if self._results:
            return self._results.popleft()

        self._loop_finished = True
        raise StopAsyncIteration  # All tasks done and all results retrieved


def result_to_str(rc: ResultCapture):
    # FIXME: Put in aioresult.ResultBase.__str__()
    if not rc.is_done():
        return "ResultCapture(not done)"
    elif rc.exception() is not None:
        return f"ResultCapture(exception: {rc.exception()!r})"
    else:
        return f"ResultCapture(result: {rc.result()})"


if __name__ == "__main__":
    import random

    async def rand(i):
        sleep_length = random.random()
        try:
            print(f"Starting {i}: {sleep_length}")
            await trio.sleep(sleep_length)
            print(f"Finished {i}: {sleep_length}")
            return sleep_length
        except BaseException:
            print(f"Exception {i}: {sleep_length}")
            raise

    async def main(count):
        async with trio.open_nursery() as outer_nursery:
            async with StreamResultsNursery(max_running_tasks=3) as N:
                for i in range(count):
                    print(f"Starting task {i}")
                    N.start_soon(rand, i)

                i = 0
                async for rc in N:
                    i += 1
                    assert isinstance(rc, ResultCapture)
                    print(f"Got {i}: {result_to_str(rc)}")
                    if i == count:
                        print(f"starting extra task")
                        N.start_soon(rand, i)

    async def loop_results(N: StreamResultsNursery):
        i = 0
        async for rc in N:
            i += 1
            assert isinstance(rc, ResultCapture)
            print(f"Got {i}: {result_to_str(rc)}")
            if i == 5:
                print(f"cancelling")
                N.cancel_scope.cancel()


    async def main_other(count):
        async with trio.open_nursery() as outer_nursery:
            async with StreamResultsNursery(max_running_tasks=math.inf) as N:
                for i in range(count):
                    print(f"Starting task {i}")
                    N.start_soon(rand, i)

                outer_nursery.start_soon(loop_results, N)


    trio.run(main,10)
    trio.run(main_other, 10)
