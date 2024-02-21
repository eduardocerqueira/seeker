#date: 2024-02-21T17:02:46Z
#url: https://api.github.com/gists/8bb7e0575572214b5e7be830dd95f7ba
#owner: https://api.github.com/users/mvandermeulen

from redis.asyncio import Redis


class PrefixRedis(Redis):

    def __init__(self, *args, prefix: str = "", **kwargs):
        self._prefix = prefix
        super().__init__(*args, **kwargs)

    def generate_key(self, key: str):
        return f"{self._prefix}:{key}" if self._prefix else key

    async def execute_command(self, *args, **options):
        # Add prefix to key
        if len(args) >= 2:
            args = list(args)
            args[1] = self.generate_key(args[1])

        return await super().execute_command(*args, **options)