#date: 2022-03-15T16:46:27Z
#url: https://api.github.com/gists/dfb43e54e51abaa47a4690f39b0e99d5
#owner: https://api.github.com/users/say-yawn

async def print_num(num):
    print(num)
    return num

async def main_2():
    tasks = []
    for num in range(10):
        tasks.append(asyncio.create_task(print_num(num)))
    numbers = await asyncio.gather(*tasks)
    print(numbers)
    print(sum(numbers))

async def main():
    numbers = []
    for num in range(10):
        numbers.append(await print_num(num))
    print(numbers)
    print(sum(numbers))

def measure_time(function):
    import time
    s = time.perf_counter()
    asyncio.run(function())
    elapsed = time.perf_counter() - s
    print(f'Executed in {elapsed:0.6f} seconds')