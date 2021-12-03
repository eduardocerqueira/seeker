#date: 2021-12-03T17:02:05Z
#url: https://api.github.com/gists/f15c55b6d0707f56d2771e90ce165abd
#owner: https://api.github.com/users/dgtlctzn

task: asyncio.Task = asyncio.ensure_future(
    get_url(session, url)
)