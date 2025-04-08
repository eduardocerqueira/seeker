#date: 2025-04-08T17:08:46Z
#url: https://api.github.com/gists/a1af32ad7e7622eee6b35421467e76d5
#owner: https://api.github.com/users/EdsonAlcala

from hashlib import sha3_256
from base58 import b58decode
from ecdsa import SECP256k1
from ecdsa.ellipticcurve import Point

from cdp_agentkit_core.actions.near.constants import EPSILON_DERIVATION_PREFIX, ROOT_PUBLIC_KEY_MAINNET, ROOT_PUBLIC_KEY_TESTNET

def get_root_public_key(network: str = "near-mainnet") -> str:
    """
    Returns the root public key according to the network.
    
    :param network: "near-mainnet" or "near-testnet" (case insensitive)
    :return: The root public key string.
    """
    net = network.lower()
    if net == "near-testnet":
        return ROOT_PUBLIC_KEY_TESTNET
    elif net == "near-mainnet":
        return ROOT_PUBLIC_KEY_MAINNET
    else:
        raise ValueError(f"Unsupported network: {network}")
    
def derive_epsilon(account_id: str, path: str) -> int:
    """
    Deterministically derive an epsilon value from a NEAR account identifier and a derivation path.

    This function constructs the following string: "near-mpc-recovery v0.1.0 epsilon derivation:<account_id>,<path>"
    and applies SHA3-256 to it. The resulting 32-byte hash is then converted to an integer.

    :param account_id: The NEAR account identifier (e.g., "chainsignature.near")
    :param path: The derivation path (e.g., "bitcoin-1" or "ethereum-1") as a string. Defaults to "ethereum-1".
    :return: The epsilon value as an integer.
    """
    derivation_input = f"{EPSILON_DERIVATION_PREFIX}{account_id},{path}"
    hash_bytes = sha3_256(derivation_input.encode("utf-8")).digest()
    epsilon = int.from_bytes(hash_bytes, byteorder="big")
    return epsilon

def derive_public_key(root_public_key_str: str, epsilon: int) -> bytes:
    """
    Derives a new public key from a root public key and an epsilon value.

    The root public key must be provided in the format:
         "secp256k1:<Base58 encoded 64 bytes>"
    
    Steps:
      1. Remove the "secp256k1:" prefix and decode the Base58 part to obtain 64 bytes.
      2. Prepend the byte 0x04 to convert it into an uncompressed public key (65 bytes).
      3. Extract the X and Y coordinates and construct the EC point on the secp256k1 curve.
      4. Compute the derived point as:
             derived_point = root_point + (epsilon * G)
         where G is the generator of the curve.
      5. Return the derived public key in uncompressed format (0x04 || X || Y).

    :param root_public_key_str: The root public key in the format "secp256k1:<Base58 encoded 64 bytes>"
    :param epsilon: The epsilon value derived by `derive_epsilon`
    :return: The derived public key as bytes (uncompressed, 65 bytes).
    """
    prefix = "secp256k1:"
    if not root_public_key_str.startswith(prefix):
        raise ValueError("Invalid root public key format. Must start with 'secp256k1:'.")

    # Extract and decode the Base58 portion.
    base58_part = root_public_key_str[len(prefix):]
    decoded = b58decode(base58_part)
    if len(decoded) != 64:
        raise ValueError("Decoded root public key must be 64 bytes long.")

    # Prepend 0x04 to obtain an uncompressed public key (65 bytes).
    uncompressed_pub = b'\x04' + decoded
    x = int.from_bytes(uncompressed_pub[1:33], byteorder="big")
    y = int.from_bytes(uncompressed_pub[33:65], byteorder="big")

    # Create the root point on the secp256k1 curve.
    curve = SECP256k1.curve
    root_point = Point(curve, x, y)

    # Get the curve generator.
    generator = SECP256k1.generator

    # Calculate the derived public key:
    # derived_point = root_point + (epsilon * generator)
    derived_point = root_point + (epsilon * generator)

    # Convert the derived point to uncompressed bytes: 0x04 || X || Y.
    x_derived = derived_point.x()
    y_derived = derived_point.y()
    derived_pub = b'\x04' + x_derived.to_bytes(32, "big") + y_derived.to_bytes(32, "big")
    return derived_pub

def get_derived_public_key(account_id: str, path: str, network: str) -> bytes:
    """
    Calculates the derived public key given a NEAR account, a derivation path and the network. 
    The function uses the constant root public key for the given network.

    :param account_id: The NEAR account identifier.
    :param path: The derivation path.
    :param network: "mainnet" or "testnet".
    :return: The derived public key as bytes (uncompressed, 65 bytes).
    """
    root_public_key_str = get_root_public_key(network)
    epsilon = derive_epsilon(account_id, path)
    return derive_public_key(root_public_key_str, epsilon)

# ------------------ Pytest Tests ------------------
def test_derive_epsilon():
    account_id = "omnitester.testnet"
    path = "bitcoin-1"
    epsilon = derive_epsilon(account_id, path)
    
    assert isinstance(epsilon, int)
    assert epsilon > 0

def test_derive_public_key_bitcoin():
    account_id = "omnitester.testnet"
    path = "bitcoin-1"
    derived_pub_bytes = get_derived_public_key(account_id, path, "testnet")
    derived_pub_hex = derived_pub_bytes.hex()

    expected_hex = (
        "0471f75dc56b971fbe52dd3e80d2f8532eb8905157556df39cb7338a67c804126"
        "40c869f717217ba5b916db6d7dc7d6a84220f8251e626adad62cac9c7d6f8e032"
    )
    assert derived_pub_hex == expected_hex


def test_derive_public_key_ethereum():
    account_id = "omnitester.testnet"
    path = "ethereum-1"
    derived_pub_bytes = get_derived_public_key(account_id, path, 'testnet')
    derived_pub_hex = derived_pub_bytes.hex()

    expected_hex = (
        "04e612e7650febebc50b448bf790f6bdd70a8a6ce3b111a1d7e72c87afe84be7"
        "76e36226e3f89de1ba3cbb62c0f3fc05bffae672c9c59d5fa8a4737b6547c64eb7"
    )
    assert derived_pub_hex == expected_hex