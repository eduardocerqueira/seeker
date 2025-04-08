#date: 2025-04-08T17:08:46Z
#url: https://api.github.com/gists/a1af32ad7e7622eee6b35421467e76d5
#owner: https://api.github.com/users/EdsonAlcala

import hashlib
from base58 import b58encode

from cdp_agentkit_core.actions.near.segwit_addr import bech32_encode, convertbits
from eth_hash.auto import keccak


def get_evm_address(public_key: bytes) -> str:
    """
    Computes the Ethereum address from an uncompressed public key.
    
    The public key is expected to be 65 bytes with a leading 0x04.
    According to Ethereum specifications, the 0x04 is dropped and the
    Keccak-256 hash is computed on the remaining 64 bytes (the X and Y coordinates).
    The last 20 bytes of the hash are used as the Ethereum address.
    
    :param public_key: Uncompressed public key (65 bytes)
    :return: Ethereum address as a hex string (prefixed with "0x")
    """
    # Drop the 0x04 prefix if present
    if public_key[0] == 0x04:
        pubkey_no_prefix = public_key[1:]
    else:
        pubkey_no_prefix = public_key

    # Calcula el hash Keccak directamente sobre los datos
    hash_bytes = keccak(pubkey_no_prefix)
    eth_address = hash_bytes[-20:]
    return "0x" + eth_address.hex()


def get_btc_legacy_address(public_key: bytes, network: str = 'bitcoin') -> str:
    """
    Computes the Bitcoin legacy (P2PKH) address from a public key.
    
    Steps:
      1. Compute SHA256, then RIPEMD160 of the public key.
      2. Prepend the version byte: 0x00 for Bitcoin mainnet, 0x6f for testnet.
      3. Compute the checksum: first 4 bytes of double SHA256 of the payload.
      4. Concatenate payload and checksum, then encode using Base58.
    
    :param public_key: Public key bytes (can be uncompressed or compressed)
    :param network: 'bitcoin' (mainnet) or 'testnet'
    :return: Base58Check encoded Bitcoin legacy address.
    """
    sha256_hash = hashlib.sha256(public_key).digest()
    ripemd160_hash = hashlib.new('ripemd160', sha256_hash).digest()

    version_byte = b'\x00' if network == 'bitcoin' else b'\x6f'
    payload = version_byte + ripemd160_hash
    # Compute checksum: first 4 bytes of double SHA256 of the payload
    checksum = hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4]
    full_payload = payload + checksum
    address = b58encode(full_payload).decode()
    return address


def get_btc_segwit_address(public_key: bytes, network: str = 'bitcoin') -> str:
    """
    Computes the Bitcoin SegWit (P2WPKH) address from a public key.
    
    Steps:
      1. Compute SHA256, then RIPEMD160 of the public key.
      2. Convert the resulting 20-byte hash (witness program) from 8-bit to 5-bit groups.
      3. Prepend the witness version (0 for P2WPKH).
      4. Encode using Bech32 with HRP: "bc" for mainnet, "tb" for testnet.
    
    :param public_key: Public key bytes.
    :param network: 'bitcoin' (mainnet) or 'testnet'
    :return: SegWit address in Bech32 format.
    """
    sha256_hash = hashlib.sha256(public_key).digest()
    ripemd160_hash = hashlib.new('ripemd160', sha256_hash).digest()
    
    witness_version = 0
    # Convert the 20-byte hash to 5-bit groups.
    converted = convertbits(list(ripemd160_hash), 8, 5, True)
    if converted is None:
        raise ValueError("Error converting hash to 5-bit groups for Bech32 encoding")
    data = [witness_version] + converted
    hrp = 'bc' if network == 'bitcoin' else 'tb'
    segwit_addr = bech32_encode(hrp, data, "bech32")
    return segwit_addr


# ------------------ Pytest Tests ------------------
def test_get_evm_address():
    public_key_hex = (
        "04e612e7650febebc50b448bf790f6bdd70a8a6ce3b111a1d7e72c87afe84be7"
        "76e36226e3f89de1ba3cbb62c0f3fc05bffae672c9c59d5fa8a4737b6547c64eb7"
    )
    public_key = bytes.fromhex(public_key_hex)
    evm_addr = get_evm_address(public_key)

    expected_evm_addr = "0xd8d25820c9b9e2aa9cce55504355e500efcce715"
    assert evm_addr == expected_evm_addr, f"Expected {expected_evm_addr}, got {evm_addr}"


def test_get_btc_legacy_address():
    public_key_hex = (
        "04e612e7650febebc50b448bf790f6bdd70a8a6ce3b111a1d7e72c87afe84be7"
        "76e36226e3f89de1ba3cbb62c0f3fc05bffae672c9c59d5fa8a4737b6547c64eb7"
    )
    public_key = bytes.fromhex(public_key_hex)
    # For testnet, the expected legacy address (Base58Check) might be known.
    # Here, as an example, we check that the address starts with "m" or "n" (typical for testnet).
    legacy_addr = get_btc_legacy_address(public_key, network='testnet')
    assert legacy_addr[0] in ('m', 'n'), f"Unexpected testnet legacy address: {legacy_addr}"


def test_get_btc_segwit_address():
    public_key_hex = (
        "04e612e7650febebc50b448bf790f6bdd70a8a6ce3b111a1d7e72c87afe84be7"
        "76e36226e3f89de1ba3cbb62c0f3fc05bffae672c9c59d5fa8a4737b6547c64eb7"
    )
    public_key = bytes.fromhex(public_key_hex)
    segwit_addr = get_btc_segwit_address(public_key, network='testnet')
    # For testnet, segwit addresses typically start with "tb1"
    assert segwit_addr.startswith("tb1"), f"Unexpected testnet segwit address: {segwit_addr}"
