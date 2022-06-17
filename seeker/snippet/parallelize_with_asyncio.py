#date: 2022-06-17T16:46:12Z
#url: https://api.github.com/gists/869656326f4f5f7fc1937eacc0bea6a9
#owner: https://api.github.com/users/gregoryfdel

import asyncio
from asyncio import Task
import nest_asyncio

from functools import partial, wraps
import time

def slow_fn(i):
    print("Called slow_fn with ", i)
    time.sleep(1.)
    return i*i+i

MAX_CONCURRENT = 80
MAX_SIZE = 80
return_list = []
fn_to_parallelize = slow_fn

def async_wrap(f):
    @wraps(f)
    async def run(*args, loop=None, executor=None, **kwargs):
        global return_list
        if loop is None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:  # no event loop running:
                loop = asyncio.new_event_loop()
        p = partial(f, *args, **kwargs)
        data = await loop.run_in_executor(executor, p)
        return_list.append(data)
    return run

async def run_coroutine(task_queue, semaphore, i_coroutine, i_args):
    async with semaphore:
        await i_coroutine(*i_args)
        task_queue.task_done()

async def run_function(arg_list):
    global return_list
    return_list = []
    
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    task_queue = asyncio.Queue(MAX_SIZE)
    
    slow_fn_async = async_wrap(fn_to_parallelize)

    async for args in arg_list:
        task = asyncio.create_task(
            run_coroutine(task_queue, semaphore, slow_fn_async, args)
        )
        await task_queue.put(task)
        if (MAX_SIZE - i_arg) < 2:
            await task_queue.join()
            task_queue = asyncio.Queue(MAX_SIZE)
            i_arg = 0
    await task_queue.join()

def _to_task(future, as_task, loop):
    if not as_task or isinstance(future, Task):
        return future
    return loop.create_task(future)        

async def desync(it):
  for x in it: yield x

def asyncio_run(future, as_task=True):

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:  # no event loop running:
        loop = asyncio.new_event_loop()
        return loop.run_until_complete(_to_task(future, as_task, loop))
    else:
        nest_asyncio.apply(loop)
        return asyncio.run(_to_task(future, as_task, loop))
    
i_args = [[ii] for ii in list(range(1,5))]
asyncio_run(run_function(desync(i_args)))

print(return_list)