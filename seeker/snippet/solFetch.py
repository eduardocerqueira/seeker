#date: 2026-02-10T17:40:30Z
#url: https://api.github.com/gists/ad800de2c0548e0dbba3bb461dcd2fe9
#owner: https://api.github.com/users/VoidCounsel

from solana.rpc.api import Client
import base64

client = Client("https://api.mainnet-beta.solana.com")

slot = client.get_slot().value
print(f"Current slot: {slot:,}")

block = client.get_block(
    slot,
    encoding="base64",
    max_supported_transaction_version=0,
).value

if not block:
    print("Block not available")
    exit()

print(f"Block slot:   {block.parent_slot + 1}")
print(f"Time:         {block.block_time}")
print(f"Hash:         {block.blockhash}")
print(f"Tx count:     {len(block.transactions)}")

print("\nFirst transaction raw info:")
tx_meta = block.transactions[0]
raw_bytes = bytes(tx_meta.transaction)
b64_str = base64.b64encode(raw_bytes).decode()

print("Signature:", tx_meta.transaction.signatures[0])
print("Length:   ", len(raw_bytes), "bytes")
print("Base64 first 80 chars:", b64_str[:80] + "...")
print("First 50 bytes (hex):", raw_bytes[:50].hex())