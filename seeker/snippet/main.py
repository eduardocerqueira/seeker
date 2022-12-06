#date: 2022-12-06T17:03:19Z
#url: https://api.github.com/gists/92cc054a94180b25accb74b87afb083c
#owner: https://api.github.com/users/mypy-play

from typing import Awaitable, Callable, Optional

class bytes32:
    ...

class CoinRecord:
    ...


class MempoolManager:
    get_coin_record: Callable[[bytes32], Awaitable[Optional[CoinRecord]]]
    
    def __init__(
         self,
        get_coin_record: Callable[[bytes32], Awaitable[Optional[CoinRecord]]],
    ):
        self.get_coin_record = get_coin_record
        #setattr(self, "get_coin_record", get_coin_record)

