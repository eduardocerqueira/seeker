#date: 2022-05-05T17:19:08Z
#url: https://api.github.com/gists/87cef2454d80e48e45b7333c935a74d9
#owner: https://api.github.com/users/ZacharyHampton

class CustomClientSession(aiohttp.ClientSession):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def _request(self, *args, **kwargs):
        if 'proxy' not in kwargs:
            with open(os.getcwd() + '/proxies.txt', 'r') as p:
                proxies = p.read().splitlines()

            proxy = proxies[random.randint(0, len(proxies) - 1)].rstrip()

            kwargs['proxy'] = "http://" + proxy
        return await super()._request(*args, **kwargs)