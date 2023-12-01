#date: 2023-12-01T16:39:59Z
#url: https://api.github.com/gists/971b8a86bc1cd87b141a640fcf4e05aa
#owner: https://api.github.com/users/Shaweeen

import json
import time

from mospy.clients import HTTPClient

import httpx
from mospy import Account, Transaction

client = HTTPClient(api="https://cosmos-rest.publicnode.com")

nonce = 0

account = Account(
    private_key='你的16进制私钥，不带0x',
)


def do():
    global nonce
    account.next_sequence = nonce
    tx = Transaction(
        account=account,
        gas=75000,
        memo=r'ZGF0YToseyJvcCI6Im1pbnQiLCJhbXQiOjEwMDAwLCJ0aWNrIjoiY29zcyIsInAiOiJjcmMtMjAifQ=='
    )
    tx.set_fee(
        amount=375,
        denom="uatom"
    )
    tx.add_msg(
        tx_type='transfer',
        sender=account,
        receipient="你的地址",
        amount=1,
        denom="uatom",
    )

    tx_bytes = tx.get_tx_bytes_as_string()
    rpc_url = "https://cosmos-rpc.publicnode.com:443"
    pushable_tx = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "broadcast_tx_sync",
            "params": {
                "tx": tx_bytes
            }
        }
    )
    r = httpx.post(rpc_url, data=pushable_tx)
    if 'hash' not in r.text:
        raise Exception("error", r.text)
    print(r.text)
    nonce += 1
    print(nonce)


while 1:
    try:
        client.load_account_data(account=account)
    except Exception as e:
        print(e)
        continue
    nonce = account.next_sequence
    try:
        for i in range(10):
            do()
            time.sleep(0.2)
    except Exception as e:
        print(e)
        time.sleep(1)
