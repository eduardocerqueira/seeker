#date: 2025-04-08T17:08:46Z
#url: https://api.github.com/gists/a1af32ad7e7622eee6b35421467e76d5
#owner: https://api.github.com/users/EdsonAlcala

from collections.abc import Callable

from cdp import Wallet
from pydantic import BaseModel, Field, field_validator

from cdp_agentkit_core.actions import CdpAction
from cdp_agentkit_core.actions.near.kdf import get_derived_public_key
from cdp_agentkit_core.actions.near.constants import DEFAULT_PATH
from cdp_agentkit_core.actions.near.supported_address_types import SUPPORTED_ADDRESS_TYPES
from cdp_agentkit_core.actions.near.address import get_evm_address, get_btc_legacy_address, get_btc_segwit_address

GET_CROSS_CHAIN_ADDRESS_PROMPT = """
This tool computes a cross chain address of a particular type using the derivation path, network, NEAR account id and the type of address, returning the result in hex string format.

The derived address is compatible with ECDSA and can be used to interact with contracts or perform transactions on the specified chain.

"""

class GetCrossChainAddressInput(BaseModel):
    """Input argument schema for get cross chain address action."""

    account_id: str | None = Field(
        None,
        description="The NEAR account id. If not provided, uses the wallet's default address.",
    )
    network: str | None = Field(
        None,
        description="The NEAR network. If not provided, uses the wallet's network id, which defaults to `near-mainnet`.",
    )
    path: str | None = Field(
        None,
        description="The derivation path to compute the public key, e.g. `Ethereum-1`. If not provided, uses the default derivation path `account-1`.",
    )
    address_type: str = Field(
        ...,
        description="The address type based on the target chain and type of address for networks like Bitcoin and Ethereum (e.g., `evm` or `bitcoin-mainnet-legacy`).",
    )

    @field_validator("address_type")
    def validate_chain(cls, value):
        """Validate the address type against the supported address types."""
        # Ensure the address type is supported
        if value not in SUPPORTED_ADDRESS_TYPES:
            raise ValueError(
                f"Unsupported address type: {value}. Supported address types are: {', '.join(SUPPORTED_ADDRESS_TYPES)}."
            )

        return value


def get_cross_chain_address(wallet: Wallet, account_id: str | None, network: str | None, path: str | None, address_type: str) -> str:
    """Computes an address for a specific chain using the account id, network, derivation path, and chain identifier.
    """
    check_account_id = account_id if account_id is not None else wallet.default_address.address_id
    check_path = path if path is not None else DEFAULT_PATH
    check_network = network if network is not None else wallet.network_id
    public_key = None
    address = None
    try:
        public_key = get_derived_public_key(check_account_id, check_path, check_network)
        match address_type:
            case "evm":
                address = get_evm_address(public_key)
            case "bitcoin-mainnet-legacy":
                address = get_btc_legacy_address(public_key, "bitcoin")
            case "bitcoin-mainnet-segwit":
                address = get_btc_segwit_address(public_key, "bitcoin")
            case "bitcoin-testnet-legacy":
                address = get_btc_legacy_address(public_key, "testnet")
            case "bitcoin-testnet-segwit":
                address = get_btc_segwit_address(public_key, "testnet")
    except Exception as e:
        return f"Error generating the address: {e!s}"

    return f"Generated cross chain address of type {address_type} for account id {account_id}, network {network} and derivation path {path} is {address}."


class GetCrossChainAddressAction(CdpAction):
    """Get Cross Chain Address action."""

    name: str = "get_cross_chain_address"
    description: str = GET_CROSS_CHAIN_ADDRESS_PROMPT
    args_schema: type[BaseModel] | None = GetCrossChainAddressInput
    func: Callable[..., str] = get_cross_chain_address
