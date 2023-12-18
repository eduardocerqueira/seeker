#date: 2023-12-18T17:09:45Z
#url: https://api.github.com/gists/c8be96eb457b0b773bdaa4f26edc0af4
#owner: https://api.github.com/users/voith

import collections
from statistics import median

from web3 import Web3

BlockData = collections.namedtuple(
    "BlockData", ["number", "timestamp", "median_gas_price"]
)

w3 = Web3(Web3.HTTPProvider("<NODE_URL>"))


def fetch_eth_price():
	# TODO: fetch price from a TWAP oralce
	return 2161.78

def fetch_block_data(sample_size):
    latest = w3.eth.get_block("latest", full_transactions=True)

    yield BlockData(
    	latest.number,
    	latest.timestamp, 
    	statistics.median([tx.gasPrice for tx in latest.transactions])
    )

    block = latest

    for _ in range(sample_size - 1):
        block = w3.eth.get_block(block.parentHash, full_transactions=True)
        yield BlockData(
        	block.number,
        	block.timestamp, 
        	statistics.median([tx.gasPrice for tx in block.transactions])
        )
        

def calculate_weighted_average_gas_price(sample_size):
	block_data = list(fetch_block_data(sample_size))
	sorted_block_data = sorted(block_data, key=lambda x: x.timestamp)
	oldest_block_number = sorted_block_data[0].number
	prev_timestamp = w3.eth.get_block(oldest_block_number - 1).timestamp
	weighted_sum = 0.0
	sum_of_weights = 0.0
	for block in sorted_block_data:
		weight = block.timestamp - prev_timestamp
		weighted_sum += weight * block.median_gas_price
		sum_of_weights += weight
		prev_timestamp = block.timestamp
	return weighted_sum / sum_of_weights


def calculate_gas_price_in_usd():
	gas_price = calculate_weighted_average_gas_price(5) #  5 blocks
	eth_price = fetch_eth_price()
	return (gas_price / 10 ** 9) * eth_price