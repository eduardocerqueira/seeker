#date: 2025-01-30T16:42:57Z
#url: https://api.github.com/gists/0561c678bc6d061acd6843ab5dbfbb2f
#owner: https://api.github.com/users/aryogesh

from coincurve import PrivateKey
from ecdsa import SigningKey, SECP256k1
from ecdsa.util import sigencode_der
from pprint import pprint
import binascii
import requests
import json
import hashlib
import base58
from typing import Dict, Any, Optional, Tuple, List

class DeSoKeyPair:
    def __init__(self, public_key: bytes, private_key: bytes):
        self.public_key = public_key
        self.private_key = private_key

def create_key_pair_from_seed_or_seed_hex(
    seed: str,
    passphrase: str,
    index: int,
    is_testnet: bool
) -> Tuple[Optional[DeSoKeyPair], Optional[str]]:
    if not seed:
        return None, "Seed must be provided"

    try:
        seed_bytes = binascii.unhexlify(seed.lower())
        if passphrase or index != 0:
            return None, "Seed hex provided, but passphrase or index params were also provided"

        privkey = PrivateKey(seed_bytes)
        pubkey = privkey.public_key
        return DeSoKeyPair(pubkey.format(), privkey.secret), None

    except binascii.Error:
        return None, "Invalid seed hex"

def pubkey_to_base58(pubkey_bytes: bytes, is_testnet: bool) -> str:
    version_byte = b'\xcd' if is_testnet else b'\x19'
    payload = version_byte + pubkey_bytes
    checksum = hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4]
    return base58.b58encode(payload + checksum).decode('utf-8')

class DeSoDexClient:
    def __init__(self, is_testnet: bool=False, seed_phrase_or_hex=None, passphrase=None, index=0, node_url=None):
        self.is_testnet = is_testnet

        desoKeyPair, err = create_key_pair_from_seed_or_seed_hex(
            seed_phrase_or_hex, passphrase, index, is_testnet,
        )
        if desoKeyPair is None:
            raise ValueError(err)
        self.deso_keypair = desoKeyPair
        self.public_key_base58 = pubkey_to_base58(desoKeyPair.public_key, is_testnet)

        if node_url is None:
            self.node_url = "https://test.deso.org" if is_testnet else "https://node.deso.org"
        else:
            self.node_url = node_url.rstrip("/")

    def mint_or_burn_tokens(
        self,
        updater_pubkey_base58check: str,
        profile_pubkey_base58check: str,
        operation_type: str,            # 'mint' or 'burn'
        coins_to_mint_or_burn_nanos: str,
        min_fee_rate_nanos_per_kb: int = 1000,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        url = f"{self.node_url}/api/v0/dao-coin"

        payload = {
            "UpdaterPublicKeyBase58Check": updater_pubkey_base58check,
            "ProfilePublicKeyBase58CheckOrUsername": profile_pubkey_base58check,
            "OperationType": operation_type,
        }

        if operation_type.lower() == "mint":
            payload["CoinsToMintNanos"] = coins_to_mint_or_burn_nanos
        elif operation_type.lower() == "burn":
            payload["CoinsToBurnNanos"] = coins_to_mint_or_burn_nanos
        else:
            raise ValueError('operation_type must be "mint" or "burn".')

        payload["MinFeeRateNanosPerKB"] = min_fee_rate_nanos_per_kb

        headers = {
            "Content-Type": "application/json",
        }
        if extra_headers:
            headers.update(extra_headers)

        resp = requests.post(url, json=payload, headers=headers)
        try:
            resp.raise_for_status()
        except requests.exceptions.HTTPError as e:
            error_json = resp.json()  # Get the error response JSON
            raise requests.exceptions.HTTPError(f"HTTP Error: {e}, Response: {error_json}")

        return resp.json()

    def submit_post(
        self,
        updater_public_key_base58check: str,
        body: str,
        parent_post_hash_hex: Optional[str] = None,
        reposted_post_hash_hex: Optional[str] = None,
        title: Optional[str] = "",
        image_urls: Optional[List[str]] = None,
        video_urls: Optional[List[str]] = None,
        post_extra_data: Optional[Dict[str, Any]] = None,
        min_fee_rate_nanos_per_kb: int = 1000,
        is_hidden: bool = False,
        in_tutorial: bool = False
) -> Dict[str, Any]:
        url = f"{self.node_url}/api/v0/submit-post"
        payload = {
            "UpdaterPublicKeyBase58Check": updater_public_key_base58check,
            "PostHashHexToModify": "",
            "ParentStakeID": parent_post_hash_hex or "",
            "RepostedPostHashHex": reposted_post_hash_hex or "",
            "Title": title or "",
            "BodyObj": {
                "Body": body,
                "ImageURLs": image_urls or [],
                "VideoURLs": video_urls or [],
            },
            "PostExtraData": post_extra_data or {"Node": "1"},
            "Sub": "",
            "IsHidden": is_hidden,
            "MinFeeRateNanosPerKB": min_fee_rate_nanos_per_kb,
            "InTutorial": in_tutorial,
        }

        headers = {"Content-Type": "application/json"}
        response = requests.post(url, json=payload, headers=headers)

        try:
            response.raise_for_status()
            response_data = response.json()

            if 'TransactionHex' in response_data:
                return self.sign_and_submit_txn(response_data)

            return response_data

        except requests.exceptions.HTTPError as e:
            error_json = response.json() if response.content else response.text
            raise ValueError(f"HTTP Error: {e}, Response: {error_json}")


    def sign_single_txn(self, unsigned_txn_hex: str) -> str:
        try:
            # Decode hex transaction to bytes
            txn_bytes = bytes.fromhex(unsigned_txn_hex)

            # Double SHA256 hash of the transaction bytes
            first_hash = hashlib.sha256(txn_bytes).digest()
            txn_hash = hashlib.sha256(first_hash).digest()


            # Create signing key from private key bytes
            signing_key = SigningKey.from_string(self.deso_keypair.private_key, curve=SECP256k1)


            # Sign the hash
            signature = signing_key.sign_digest(txn_hash, sigencode=sigencode_der)

            # Convert signature to hex
            signature_hex = signature.hex()

            return signature_hex

        except Exception as e:
            return None

    def submit_txn(self, unsigned_txn_hex: str, signature_hex: str) -> dict:
        submit_url = f"{self.node_url}/api/v0/submit-transaction"

        payload = {
            "UnsignedTransactionHex": unsigned_txn_hex,
            "TransactionSignatureHex": signature_hex
        }

        headers = {
            "Origin": self.node_url,
            "Content-Type": "application/json"
        }

        response = requests.post(
            submit_url,
            data=json.dumps(payload),
            headers=headers
        )

        if response.status_code != 200:
            print(f"Error status returned from {submit_url}: {response.status_code}, {response.text}")
            raise ValueError(
                f"Error status returned from {submit_url}: "
                f"{response.status_code}, {response.text}"
            )

        return response.json()


    def submit_atomic_txn(
            self,
            incomplete_atomic_txn_hex: str,
            unsigned_inner_txn_hexes: List[str],
            txn_signatures_hex: List[str]
    ) -> Dict[str, Any]:
        endpoint = "/api/v0/submit-atomic-transaction"
        url = f"{self.node_url}{endpoint}"

        payload = {
            "IncompleteAtomicTransactionHex": incomplete_atomic_txn_hex,
            "UnsignedInnerTransactionsHex": unsigned_inner_txn_hexes,
            "TransactionSignaturesHex": txn_signatures_hex
        }

        response = requests.post(url, json=payload)

        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            try:
                error_json = response.json()
            except ValueError:
                error_json = response.text
            raise requests.exceptions.HTTPError(
                f"Error status returned from {url}: {response.status_code}, {error_json}"
            )

        return response.json()

    def sign_and_submit_txn(self, resp: Dict[str, Any]) -> Dict[str, Any]:
        unsigned_txn_hex = resp.get('TransactionHex')
        if unsigned_txn_hex is None:
            raise ValueError("TransactionHex not found in response")
        if 'InnerTransactionHexes' in resp:
            unsigned_inner_txn_hexes = resp.get('InnerTransactionHexes')
            signature_hexes = []
            for unsigned_inner_txn_hex in unsigned_inner_txn_hexes:
                signature_hex = self.sign_single_txn(unsigned_inner_txn_hex)
                signature_hexes.append(signature_hex)
            return self.submit_atomic_txn(
                unsigned_txn_hex, unsigned_inner_txn_hexes, signature_hexes
            )
        signature_hex = self.sign_single_txn(unsigned_txn_hex)
        return self.submit_txn(unsigned_txn_hex, signature_hex)

    def coins_to_base_units(self, coin_amount: float, is_deso: bool, hex_encode: bool = False) -> str:
        if is_deso:
            base_units = int(coin_amount * 1e9)
        else:
            base_units = int(coin_amount * 1e18)
        if hex_encode:
            return hex(base_units)
        return str(base_units)


def main():
    # Configuration
    SEED_HEX = "7123adba435ac90fd189ab4de372de13dd050186dcb292bc9f7661a4f1afe96e"
    IS_TESTNET = False
    NODE_URL = "https://test.deso.org" if IS_TESTNET else "https://node.deso.org"
    explorer_link = "https://testnet.deso.org" if IS_TESTNET else "https://deso.org"
    
    # Initialize the client
    client = DeSoDexClient(
        is_testnet=IS_TESTNET,
        seed_phrase_or_hex=SEED_HEX,
        node_url=NODE_URL
    )
    
    # Your public key (replace with actual)
    string_pubkey = "BC1YLiWM4w3BSHL781wqAbFWWuQndQbZLswk2Rt1VfLZxyWPGoNjsr1"

    print("\n ---- Mint Tokens (sign & submit - requires profile) ----")

    try:
        print('Balance before minting:')
        print('Constructing txn...')
        coins_to_mint = client.coins_to_base_units(1.0, is_deso=False, hex_encode=True)
        mint_response = "**********"
            updater_pubkey_base58check=string_pubkey,
            profile_pubkey_base58check= "**********"
            operation_type="mint",
            coins_to_mint_or_burn_nanos=coins_to_mint,
        )
        print('Txn constructed. The txn construction response often has useful information in it. Comment it in if you want to see it.')
        pprint(mint_response)
        print('Signing and submitting txn...')
        submitted_txn_response = client.sign_and_submit_txn(mint_response)
        txn_hash = submitted_txn_response['TxnHashHex']
        print(f'Waiting for commitment... Hash = {txn_hash}. Find on {explorer_link}/txn/{txn_hash}. Sometimes it takes a minute to show up on the block explorer.')
        # client.wait_for_commitment_with_timeout(txn_hash, 30.0)
        # print('Balance after minting:')
        print('SUCCESS!')

    except Exception as e:
        print(f"ERROR: "**********": {e}")

if __name__ == "__main__":
    main()t tokens call failed: "**********"

if __name__ == "__main__":
    main()