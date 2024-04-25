#date: 2024-04-25T17:00:45Z
#url: https://api.github.com/gists/07a9ffed43cf33b7e2e25fef0e5b49e6
#owner: https://api.github.com/users/DavidMinarsch

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2023 Valory AG
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# ------------------------------------------------------------------------------

from web3 import Web3
import requests
import json
import os
from decimal import Decimal, ROUND_DOWN
from itertools import chain
import csv

from dotenv import load_dotenv

load_dotenv()

ALCHEMY_API_KEY = os.getenv("ALCHEMY_API_KEY")
ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY")

w3 = Web3(Web3.HTTPProvider(f'https://eth-mainnet.g.alchemy.com/v2/{ALCHEMY_API_KEY}'))

def get_abi(contract_address):
    # Replace 'YourEtherscanApiKey' with your actual Etherscan API key
    url = f"https://api.etherscan.io/api?module=contract&action=getabi&address={contract_address}&apikey={ETHERSCAN_API_KEY}"
    response = requests.get(url)
    data = response.json()
    if data['status'] == '1':
        return json.loads(data['result'])
    else:
        raise ValueError(data['result'])

def div_evm(numerator, denominator):
    # res = numerator/denominator
    if denominator == 0:
        raise ValueError(denominator)
    return (Decimal(numerator)/Decimal(denominator)).to_integral(ROUND_DOWN)

print("------ Rewards for Devs Getting Donations in Previous Epoch Over All Previous Epochs ------")
tokenomics_implementation = "**********"
tokenomics_proxy = "**********"
abi = "**********"
tokenomics_contract = "**********"=tokenomics_proxy, abi=abi)

treasury = '0xa0DA53447C0f6C4987964d8463da7e6628B30f82'
abi = get_abi(treasury)
treasury_contract = w3.eth.contract(address=treasury, abi=abi)

component_registry = '0x15bd56669F57192a97dF41A2aa8f4403e9491776'
abi = get_abi(component_registry)
component_registry_contract = w3.eth.contract(address=component_registry, abi=abi)

agent_registry = '0x2F1f7D38e4772884b88f3eCd8B6b9faCdC319112'
abi = get_abi(agent_registry)
agent_registry_contract = w3.eth.contract(address=agent_registry, abi=abi)

service_registry = '0x48b6af7B12C71f09e2fC8aF4855De4Ff54e775cA'
abi = get_abi(service_registry)
service_registry_contract = w3.eth.contract(address=service_registry, abi=abi)

current_epoch_counter = "**********"
print(f'Epoch: {current_epoch_counter}')

# Contract inception date
from_block = 16699195
# Get donation events
event_filter = treasury_contract.events.DonateToServicesETH.create_filter(fromBlock=from_block, toBlock="latest")

# Get all entries
entries = event_filter.get_all_entries()

# Get all the component and agent Ids
component_ids = []
agent_ids = []
for entry in entries:
    service_ids = entry.args['serviceIds']
    for service_id in service_ids:
        component_ids.append(service_registry_contract.functions.getUnitIdsOfService(0, service_id).call()[1])
        agent_ids.append(service_registry_contract.functions.getUnitIdsOfService(1, service_id).call()[1])

# Flatten component and agent Ids
flattened_component_ids = list(chain.from_iterable(component_ids))
flattened_agent_ids = list(chain.from_iterable(agent_ids))

# Make unique lists
component_ids = list(set(flattened_component_ids))
agent_ids = list(set(flattened_agent_ids))
# Assemble a nested list of unique component and agent ids
unit_ids = [component_ids, agent_ids]

# Initialize dictionaries to keep track of rewards and top-ups by owner
rewards_by_owner = {}
top_ups_by_owner = {}

# Open a CSV file for writing
with open('rewards.csv', 'w', newline='') as csvfile:
    fieldnames = ['Type', 'ID', 'Rewards', 'Top-ups', 'Owner']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    # Get all finalized and pending rewards in a current epoch
    for i in range(2):
        for unit_id in unit_ids[i]:
            # Get all the unit incentives up to the current epoch
            owner = 0
            if i == 0:
                owner = component_registry_contract.functions.ownerOf(unit_id).call()
            else:
                owner = agent_registry_contract.functions.ownerOf(unit_id).call()
            incentives = "**********"

            if i == 0:
                writer.writerow({'Type': 'Component', 'ID': unit_id, 'Rewards': incentives[0], 'Top-ups': incentives[1], 'Owner': owner})
            else:
                writer.writerow({'Type': 'Agent', 'ID': unit_id, 'Rewards': incentives[0], 'Top-ups': incentives[1], 'Owner': owner})

print(f"Rewards data for Epoch {current_epoch_counter} has been exported to rewards.csv")
writerow({'Type': 'Agent', 'ID': unit_id, 'Rewards': incentives[0], 'Top-ups': incentives[1], 'Owner': owner})

print(f"Rewards data for Epoch {current_epoch_counter} has been exported to rewards.csv")
