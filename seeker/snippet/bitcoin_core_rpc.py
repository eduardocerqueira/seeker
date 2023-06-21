#date: 2023-06-21T16:49:05Z
#url: https://api.github.com/gists/0e84e6c1b21b37c87229504d5884446a
#owner: https://api.github.com/users/RGGH

#from bitcoinrpc.authproxy import AuthServiceProxy, JSONRPCException

import json
import requests
import os
from dotenv import load_dotenv
from pprint import pprint

load_dotenv()

user = os.getenv("username")
password = "**********"

# Bitcoin Core RPC information
rpc_user = user
rpc_password = "**********"
rpc_port = 8332

# Create HTTP Basic Authentication headers
rpc_auth = "**********"
rpc_url = f"http://localhost:{rpc_port}"


def get_block_info():
    payload = {
        "method": "getblockchaininfo",
        "params": [],
        "jsonrpc": "2.0",
        "id": 1,
    }
    response = requests.post(
        rpc_url,
        headers={"content-type": "application/json"},
        auth=rpc_auth,
        data=json.dumps(payload),
    )
    if response.status_code == 200:
        result = response.json()
        block_count = result["result"]["blocks"]
        block_hash = result["result"]["bestblockhash"]
        block_info =result["result"] 
        return block_info


def get_block_hash(block_height):
    payload = {
        "method": "getblockhash",
        "params": [block_height],
        "jsonrpc": "2.0",
        "id": 1,
    }
    response = requests.post(
        rpc_url,
        headers={"content-type": "application/json"},
        auth=rpc_auth,
        data=json.dumps(payload),
    )
    if response.status_code == 200:
        result = response.json()
        block_hash = result["result"]
        return block_hash
    else:
        print("Error connecting to the Bitcoin full node.")
        return None


def get_block_height():
    payload = {
        "method": "getblockcount",
        "params": [],
        "jsonrpc": "2.0",
        "id": 1,
    }
    response = requests.post(
        rpc_url,
        headers={"content-type": "application/json"},
        auth=rpc_auth,
        data=json.dumps(payload),
    )
    if response.status_code == 200:
        result = response.json()
        block_height = result["result"]
        return block_height
    else:
        print("Error connecting to the Bitcoin full node.")
        return None


block_height = get_block_height()
# print(f"{block_height}\n")
# print(get_block_hash(block_height))
print(get_block_info())
et_block_info())
