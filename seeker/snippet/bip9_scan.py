#date: 2025-12-11T16:48:55Z
#url: https://api.github.com/gists/d0d7a3db6fa959771c9204f34b5a6aa1
#owner: https://api.github.com/users/djkazic

#!/usr/bin/env python3
import requests
import json
from base64 import b64encode

RPC_USER = "bitcoin"
RPC_PASSWORD = "**********"
RPC_HOST = "127.0.0.1"
RPC_PORT = 8332

# --- BIP-110 params ---
START_HEIGHT = 927360
BIP110_BIT = 4


class BitcoinRPC:
 "**********"  "**********"  "**********"  "**********"  "**********"d "**********"e "**********"f "**********"  "**********"_ "**********"_ "**********"i "**********"n "**********"i "**********"t "**********"_ "**********"_ "**********"( "**********"s "**********"e "**********"l "**********"f "**********", "**********"  "**********"u "**********"s "**********"e "**********"r "**********", "**********"  "**********"p "**********"a "**********"s "**********"s "**********"w "**********"o "**********"r "**********"d "**********", "**********"  "**********"h "**********"o "**********"s "**********"t "**********"= "**********"" "**********"1 "**********"2 "**********"7 "**********". "**********"0 "**********". "**********"0 "**********". "**********"1 "**********"" "**********", "**********"  "**********"p "**********"o "**********"r "**********"t "**********"= "**********"8 "**********"3 "**********"3 "**********"2 "**********") "**********": "**********"
        self.url = f"http://{host}:{port}"
        self.session = requests.Session()
        auth_str = f"{user}: "**********"
        self.headers = {
            "content-type": "application/json",
            "authorization": "Basic " + b64encode(auth_str).decode(),
        }

    def call(self, method, params=None, rpc_id=1):
        if params is None:
            params = []
        payload = json.dumps(
            {
                "method": method,
                "params": params,
                "id": rpc_id,
            }
        )
        r = self.session.post(self.url, headers=self.headers, data=payload)
        r.raise_for_status()
        data = r.json()
        if data.get("error") is not None:
            raise RuntimeError(f"RPC error: {data['error']}")
        return data["result"]


def get_bip9_bits(version: int):
    """
    Return list of BIP9 signal bits (0-28) set in 'version',
    or None if 'version' is not BIP9-style.
    BIP9 requires top 3 bits (29-31) to be 001.
    """
    # Mask for top 3 bits:
    # 0xe0000000 = 1110 0000 0000 ...
    # 0x20000000 = 0010 0000 0000 ...
    if (version & 0xE0000000) != 0x20000000:
        return None

    bits = []
    for bit in range(0, 29):
        if version & (1 << bit):
            bits.append(bit)
    return bits


def main():
    rpc = "**********"

    tip_height = rpc.call("getblockcount")
    if START_HEIGHT > tip_height:
        print(f"Start height {START_HEIGHT} is above tip {tip_height}, nothing to do.")
        return

    print(f"# Scanning heights {START_HEIGHT}..{tip_height}")
    print("# Only printing blocks whose version is BIP9-style (top bits 001).")
    print(
        "# Columns: height, hash, version_dec, version_hex, bip9_bits, bip110_signals"
    )
    print(
        "# --------------------------------------------------------------------------"
    )

    bip9_block_count = 0
    bip110_signal_count = 0

    for height in range(START_HEIGHT, tip_height + 1):
        block_hash = rpc.call("getblockhash", [height])
        header = rpc.call("getblockheader", [block_hash])
        version = header["version"]

        bits = get_bip9_bits(version)
        if bits is None:
            # Not BIP9-style version; skip printing
            continue

        bip9_block_count += 1
        signals_bip110 = BIP110_BIT in bits
        if signals_bip110:
            bip110_signal_count += 1

        bits_str = ",".join(str(b) for b in bits)
        print(
            f"{height},{block_hash},{version},{hex(version)},{bits_str},{signals_bip110}"
        )

        if (height - START_HEIGHT) % 1000 == 0 and height != START_HEIGHT:
            done = height - START_HEIGHT
            total = tip_height - START_HEIGHT + 1
            pct = done * 100.0 / total
            print(f"# Progress: {done}/{total} blocks ({pct:.2f}%)", flush=True)

    print(
        "# --------------------------------------------------------------------------"
    )
    print(f"# BIP9-style blocks in range: {bip9_block_count}")
    print(f"# Blocks signaling BIP-110 (bit {BIP110_BIT}): {bip110_signal_count}")


if __name__ == "__main__":
    main()