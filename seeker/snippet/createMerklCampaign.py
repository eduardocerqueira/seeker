#date: 2025-07-30T16:49:10Z
#url: https://api.github.com/gists/76347782ca7d5d0c2faf91e76aab3e1e
#owner: https://api.github.com/users/BaptistG

import requests
import json
from web3 import Web3

RPC_URL = "RPC"
CHAIN_ID = 999
CREATOR_ADDRESS = "0xB0F6e8f4bE7CbCAD2FC107471e7F6C9d37e742Ef"
CREATOR_ADDRESS_PRIVATE_KEY = "XXX"

w3 = Web3(Web3.HTTPProvider(RPC_URL))

# ABI can be found here: https://hyperevmscan.io/address/0x7db28175b63f154587bbb1cae62d39ea80a23383#code
with open('CampaignCreator.json') as f:
    campaign_creator_abi = json.load(f)

# Standard ERC20 ABI for token allowance
with open('erc20_abi.json') as f:
    erc20_abi = json.load(f)

campaignCreator = w3.eth.contract(address='0x8BB4C975Ff3c250e0ceEA271728547f3802B36Fd', abi=campaign_creator_abi)

targetToken = "**********"
amount = 250_000
startTimestamp = 1751968800  # Example start timestamp
endTimestamp = 1752573600    # Example end timestamp
url = "https://app.hyperbeat.org/vaults/usdt"

payload = {
    "campaignType": 18, # DO NOT CHANGE THIS
    "computeChainId": CHAIN_ID,
    "distributionChainId": CHAIN_ID,
    "hooks": [],
    "targetToken": "**********"
    # Add the 18 decimals to the amount
    "amount": f"{int(amount)}000000000000000000",
    "startTimestamp": startTimestamp,
    "endTimestamp": endTimestamp,
    "creator": "0xB0F6e8f4bE7CbCAD2FC107471e7F6C9d37e742Ef",
    "rewardToken": "**********"
    "url": url,
    "whitelist": [],
    "blacklist": []
}

response = requests.post("https://api.merkl.xyz/v4/payload", json=payload)

# Print in pretty JSON format with indentation
if response.status_code == 200:
    data = response.json()
    args = data["solidity"]["args"]

    # Give an allowance to the Merkl contract (This is not optimal, you could run one allowance for all campaign or just give infinite allowance once)
    token_contract = "**********"=targetToken, abi=erc20_abi)
    allowance_tx = "**********"
        '0x8BB4C975Ff3c250e0ceEA271728547f3802B36Fd',
        int(args['amount'])
    ).build_transaction({
                    'from': CREATOR_ADDRESS,
                    'nonce': w3.eth.get_transaction_count(CREATOR_ADDRESS),
                    'chainId': CHAIN_ID
                })
    allowance_tx_signed = w3.eth.account.sign_transaction(allowance_tx, CREATOR_ADDRESS_PRIVATE_KEY)
    allowance_tx_hash = w3.eth.send_raw_transaction(allowance_tx_signed.rawTransaction)
    w3.eth.wait_for_transaction_receipt(allowance_tx_hash)
    print(f'Allowance set, tx: {allowance_tx_hash.hex()}')

    tx = campaignCreator.functions.createCampaign(
        "0x0000000000000000000000000000000000000000000000000000000000000000",
        args['creator'],
        args['rewardToken'],
        args['amount'],
        args['campaignType'],
        args['startTimestamp'],
        args['duration'],
        args['campaignData']
    ).build_transaction({
                    'from': CREATOR_ADDRESS,
                    'nonce': w3.eth.get_transaction_count(CREATOR_ADDRESS),
                    'chainId': CHAIN_ID
                })
    
    tx_signed = w3.eth.account.sign_transaction(tx, CREATOR_ADDRESS_PRIVATE_KEY)
    tx_hash = w3.eth.send_raw_transaction(tx_signed.rawTransaction)
    w3.eth.wait_for_transaction_receipt(tx_hash)
    print(f'Campaign Created, tx: {tx_hash.hex()}')
else:
    print(f"Error: {response.status_code}")
    print(response.text)
    print(json.dumps(response.json(), indent=4))  # Print the error response in pretty
status_code}")
    print(response.text)
    print(json.dumps(response.json(), indent=4))  # Print the error response in pretty
