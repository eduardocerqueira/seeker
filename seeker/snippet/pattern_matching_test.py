#date: 2023-04-27T17:04:48Z
#url: https://api.github.com/gists/a5786d06df8b945f957afe00d3464f20
#owner: https://api.github.com/users/tsuji-tomonori

from typing import Any


def check(res: dict[str, Any]) -> None:
    """
    DynamoDBのレコードをgetした結果のうちpkey, value がある場合その値を表示する.
    """
    match res:
        case {"Item": {"pkey": pkey, "value": value}}:
            print(f"pkey is {pkey}, value is {value}")
        case _:
            print("no item")


# pkey is test, value is hoge
check({"Item": {"pkey": "test", "value": "hoge"}, "ConsumedCapacity": "nanika"})
# no item
# エラーは発生しない
check({"Item": {"pkey": "test"}, "ConsumedCapacity": "nanika"})
# no item
# エラーは発生しない
check({"Item": {"value": "hoge"}, "ConsumedCapacity": "nanika"})
# no item
check({"ConsumedCapacity": "nanika"})
