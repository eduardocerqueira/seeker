#date: 2025-04-21T16:33:13Z
#url: https://api.github.com/gists/16e89122ca5c9a21a9e61a0d60b9e2ab
#owner: https://api.github.com/users/monological

#!/usr/bin/env python3

# pip install scapy solana

import sys
from scapy.all import rdpcap, UDP, IP, Raw, wrpcap, Ether
from solders.transaction import Transaction

from base64 import b64encode

if len(sys.argv) < 3:
    print("Usage: decode_tx.py <pcap_file> <output_file>")
    sys.exit(1)

packets = rdpcap(sys.argv[1])
first_printed = False

fixed = []

for i, pkt in enumerate(packets):
    if not UDP in pkt:
        print(f"Packet {i}: ❌ no UDP layer")
        continue

    raw = bytes(pkt[UDP].payload)
    try:
        tx = Transaction.from_bytes(raw)
        tx.verify()

        print(f"✅ Packet {i}: {len(tx.signatures)} signature(s), signature valid: ✅")

        if not first_printed:
            print("\nFirst valid transaction:")
            print("  - Base64:", b64encode(bytes(tx)).decode())
            print("  - Signatures:")
            for sig in tx.signatures:
                print("    •", str(sig))
            print("  - Account keys:")
            for key in tx.message.account_keys:
                print("    •", str(key))
            print("  - Program IDs:")
            for ix in tx.message.instructions:
                prog_idx = ix.program_id_index
                print("    •", str(tx.message.account_keys[prog_idx]))
            first_printed = True

        pkt = Ether() / IP(src="1.2.3.4", dst="5.6.7.8") / UDP(sport=12345, dport=8003) / Raw(load=tx)

        fixed.append(pkt)


    except Exception as e:
        print(f"❌ Packet {i}: Invalid - {e}")

wrpcap(sys.argv[2], fixed)
print(f"Fixed {len(fixed)} packets and saved to {sys.argv[2]}")