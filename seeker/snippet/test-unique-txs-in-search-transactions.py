#date: 2025-04-07T17:11:04Z
#url: https://api.github.com/gists/115ff59911c70ba9df419cbb1f96935f
#owner: https://api.github.com/users/linconvidal

import json
import http.client

# Base URL components
host = 'localhost'
port = 8082
base_path = ''

headers = {
    'Content-Type': 'application/json',
    'Accept': 'application/json'
}

def fetch_transactions(address, limit=10):
    """
    Fetches transactions from the Cardano blockchain using the search/transactions endpoint.
    Tracks unique transactions and detects duplicates.
    
    Args:
        address (str): The address to search for transactions
        limit (int): Number of transactions to fetch per page (default: 10)
        
    Returns:
        list: A list of unique transactions
    """
    offset = 0
    all_transactions = []
    seen_tx_hashes = set()
    # Track duplicates by storing tx_hash -> offsets mapping
    duplicates = {}
    page = 1

    while True:
        print(f"\nFetching page {page} with offset {offset}")
        data = {
            "network_identifier": {
                "blockchain": "cardano",
                "network": "mainnet"
            },
            "account_identifier": {
                "address": address
            },
            "offset": offset,
            "limit": limit
        }

        conn = http.client.HTTPConnection(host, port)
        conn.request("POST", base_path + "/search/transactions", json.dumps(data), headers)

        response = conn.getresponse()
        if response.status != 200:
            print(f"Request failed with status code {response.status}")
            conn.close()
            break

        result = json.loads(response.read().decode())
        conn.close()

        transactions = result.get("transactions", [])
        total_count = result.get("total_count", 0)
        next_offset = result.get("next_offset")
        
        # Print the actual response values for pagination
        print(f"API Response - total_count: {total_count}, next_offset: {next_offset}")

        # Track unique and duplicate counts for this page
        unique_in_page = 0
        duplicates_in_page = 0
        
        for tx in transactions:
            tx_hash = tx['transaction']['transaction_identifier']['hash']
            
            if tx_hash not in seen_tx_hashes:
                # New unique transaction
                seen_tx_hashes.add(tx_hash)
                all_transactions.append(tx)
                unique_in_page += 1
            else:
                # Duplicate transaction found!
                duplicates_in_page += 1
                if tx_hash in duplicates:
                    duplicates[tx_hash].append(offset)
                else:
                    # First time seeing this duplicate, record its original offset
                    # Find when we first saw this transaction
                    for i, prev_tx in enumerate(all_transactions):
                        if prev_tx['transaction']['transaction_identifier']['hash'] == tx_hash:
                            # Calculate the original offset based on position in all_transactions
                            first_offset = (i // limit) * limit
                            duplicates[tx_hash] = [first_offset, offset]
                            break

        print(f"Page {page}: Fetched {len(transactions)} transactions - {unique_in_page} unique, {duplicates_in_page} duplicates")
        print(f"Total unique transactions so far: {len(all_transactions)}")
        
        # Report any duplicates found in this page
        if duplicates_in_page > 0:
            print("⚠️  DUPLICATES DETECTED IN THIS PAGE:")
            for tx_hash, offsets in duplicates.items():
                if offset in offsets:  # Only show duplicates found in current page
                    print(f"  - Transaction {tx_hash[:8]}... was seen at offsets: {offsets}")

        if total_count == 0:
            print(f"Stopping: total_count is 0")
            break

        offset = next_offset
        page += 1

    # Summary of all duplicates
    if duplicates:
        print("\n🚨 DUPLICATE TRANSACTIONS SUMMARY:")
        print(f"Found {len(duplicates)} unique transaction hashes that appeared multiple times")
        for tx_hash, offsets in duplicates.items():
            print(f"  - {tx_hash}: appeared at offsets {offsets}")
    else:
        print("\n✅ NO DUPLICATE TRANSACTIONS FOUND. API working correctly!")

    return all_transactions


# Random address with 106 transactions according to Cardanoscan, but just 17 unique according to this script
#address = "addr1v92ecw588ply6nwk35lczw9nh48u044sxqkwzwfmz4tcneqff35jr"

# Random address with 1966 transactions according to Cardanoscan, but just 197-198 unique according to this script, 
address = "addr1qx83r3rm86g6uzl0hy2y66vf9tena4jv2hwx9fm0z9qlpcmluuch2kt7ya2mlzxmqdy3suqcx4fmnhfyq5m4wnp0vf8qrug03k"
# 2 duplicates found:
#  - c44f42ccaf47bdbfbfa4ae3b4de922e4cea67debb86f84b134f3d98c85f4589d: appeared at offsets [150, 170]
#  - 594d6faa0ad9ac0be0008d40934965da38dc7c929c23cbe3c9babbd3137ac54b: appeared at offsets [70, 180]
#  - 342e83ce742495e6e869fb3f2d37a1ee34020c50370b22c2cd06a936a737abe9: appeared at offsets [40, 190] (appears sometimes)
transactions = fetch_transactions(address)

print(f"\nFinal count of all unique transactions: {len(transactions)}")