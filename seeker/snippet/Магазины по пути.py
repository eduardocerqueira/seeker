#date: 2024-04-23T17:09:35Z
#url: https://api.github.com/gists/26681ce933713dd8fb2f8fd64c57abce
#owner: https://api.github.com/users/AspirantDrago

import os
from dataclasses import dataclass
import time
import asyncio

TIME_COEFF = 100


@dataclass
class Stop:
    duration: int
    time_in_road: int


@dataclass
class Present:
    name: str
    time_to_select: int
    time_to_buy: int

    def total_time(self) -> int:
        return self.time_to_buy + self.time_to_select

    async def buy(self) -> None:
        time.sleep(self.time_to_select / TIME_COEFF)
        print('Buy ' + self.name)
        await asyncio.sleep(self.time_to_buy / TIME_COEFF)
        print('Got ' + self.name)


async def main():
    stops: list[Stop] = []
    presents: list[Present] = []
    while s := input():
        t1, t2 = map(int, s.split())
        stops.append(Stop(t1, t2))
    while s := input():
        name, t1, t2 = s.split()
        t1, t2 = map(int, (t1, t2))
        presents.append(Present(name, t1, t2))

    for i, stop in enumerate(stops, 1):
        print(f'Buying gifts at {i} stop')
        tasks = []
        while True:
            ind_cur_present = -1
            for ind, present in enumerate(presents):
                if present.total_time() <= stop.duration:
                    if ind_cur_present == -1 or present.total_time() > presents[ind_cur_present].total_time():
                        ind_cur_present = ind
            if ind_cur_present == -1:
                break
            present = presents.pop(ind_cur_present)
            stop.duration -= present.time_to_select
            task = asyncio.create_task(present.buy())
            tasks.append(task)
        await asyncio.gather(*tasks)
        time.sleep(stop.duration / TIME_COEFF)
        print(f'Arrive from {i} stop')
        time.sleep(stop.time_in_road / TIME_COEFF)
    if presents:
        print('Buying gifts after arrival')
        while presents:
            await presents.pop().buy()


if os.name == 'nt':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
asyncio.run(main())
