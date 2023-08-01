#date: 2023-08-01T16:46:56Z
#url: https://api.github.com/gists/9c9d627a5d52012f2111f1a2059d7a2b
#owner: https://api.github.com/users/thomivy

import json
import base58
import pandas as pd
from web3 import Web3

def substrate_to_evm_address(substrate_address):
    # Decode the SS58-encoded Substrate address to bytes
    bytes_address = base58.b58decode(substrate_address)
    # The first byte is the SS58 prefix. Remove it.
    bytes_address_without_prefix = bytes_address[1:]
    # Perform the keccak-256 hash on the bytes address
    keccak_hash = Web3.keccak(bytes_address_without_prefix)
    # Take the last 20 bytes (40 characters) of the keccak hash as the EVM address
    evm_address = '0x' + keccak_hash.hex()[-40:]
    return evm_address

def convert_json_file(input_json_file_path, output_json_file_path, exclusion_csv_path, test_output_path):
    # Load the exclusion list from the CSV file
    exclusion_list = pd.read_csv(exclusion_csv_path)
    # Get the addresses from the first column (change this if the addresses are in a different column)
    excluded_addresses = set(exclusion_list.iloc[:, 0].tolist())

    with open(input_json_file_path, 'r') as file:
        data = json.load(file)

    new_data = {}
    for key, value in data.items():
        # Skip addresses that are in the exclusion list
        if key in excluded_addresses:
            continue
        try:
            new_key = substrate_to_evm_address(key)
            new_data[new_key] = value
        except ValueError:
            print(f"Invalid Substrate address format: {key}")
        
    with open(output_json_file_path, 'w') as file:
        json.dump(new_data, file, indent=4)

    # Convert excluded addresses to EVM format and check against output data
    test_results = {}
    for address in excluded_addresses:
        try:
            evm_address = substrate_to_evm_address(address)
            test_results[evm_address] = evm_address in new_data
        except ValueError:
            print(f"Invalid Substrate address format: {address}")

    # Write test results to a new file
    with open(test_output_path, 'w') as file:
        json.dump(test_results, file, indent=4)

if __name__ == "__main__":
    input_json_file_path = "snapshot.json"  # Replace with your input JSON file path
    output_json_file_path = "converted-addresses.json"  # Replace with your output JSON file path
    exclusion_csv_path = "excluded-addresses.csv"  # Replace with your exclusion CSV file path
    test_output_path = "test-results.json"  # Replace with your test output JSON file path
    convert_json_file(input_json_file_path, output_json_file_path, exclusion_csv_path, test_output_path)

