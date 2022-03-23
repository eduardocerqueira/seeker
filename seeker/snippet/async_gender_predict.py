#date: 2022-03-23T17:11:42Z
#url: https://api.github.com/gists/5605b1cb9eb2713ce473e2dda27a6006
#owner: https://api.github.com/users/anandhu-gopi

import asyncio
from typing import List
from aiohttp import ClientSession


async def predict_gender(name, session):
    params = {"name": name}
    # Fetch response from genderize asynchronously
    async with session.get("https://api.genderize.io/", params=params) as resp:
        data = await resp.json()  # read response asynchronously
        return (name, data["gender"])


async def main(names: List[str]):

    # ClientSession: Client session is the
    # recommended interface for making HTTP
    # requests (aiohttp recommendation), Session needs
    # to be closed after using it and closing session is an
    # asynchronous operation that's why we are using
    # async with (async + with)
    async with ClientSession(raise_for_status=True) as session:
        # creating gender predicting tasks
        predict_gender_tasks = [predict_gender(name, session) for name in names]

        gender_results = []
        # asyncio.as_completed(): to get tasks as they are
        # completed, in the order of completion
        for task in asyncio.as_completed(predict_gender_tasks):
            gender_result = await task
            gender_results.append(gender_result)

    print(dict(gender_results))


if __name__ == "__main__":
    names = ["Sophia", "Liam", "Olivia", "Noah"]
    # for getting event loop and running the tasks
    asyncio.run(main(names))
