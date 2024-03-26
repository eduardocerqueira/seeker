#date: 2024-03-26T17:08:42Z
#url: https://api.github.com/gists/2600812dec65cb762a2a29d3b385058b
#owner: https://api.github.com/users/sunfkny

from typing import Generic, TypeVar, cast

import eth_abi
import web3
from eth_abi.exceptions import DecodingError
from eth_typing import ChecksumAddress, HexStr
from eth_utils.abi import function_signature_to_4byte_selector
from eth_utils.address import to_checksum_address
from hexbytes import HexBytes
from typing_extensions import ParamSpec
from web3.contract.utils import ACCEPTABLE_EMPTY_STRINGS
from web3.exceptions import BadFunctionCallOutput
from web3.middleware.geth_poa import geth_poa_middleware
from web3.types import TxParams, Wei


TInput = ParamSpec("TInput")
TOut = TypeVar("TOut")


def to_web3(rpc_url: str):
    w3 = web3.Web3(web3.HTTPProvider(rpc_url))
    w3.middleware_onion.inject(geth_poa_middleware, layer=0)
    return w3


class Function(Generic[TInput, TOut]):
    def __init__(
        self,
        contract_address: ChecksumAddress,
        signature: str,
        types_input: list[str],
        types_output: list[str],
        w3: web3.Web3,
    ) -> None:
        self.w3 = w3
        self.contract_address = contract_address
        self.signature = signature
        self.selector = function_signature_to_4byte_selector(signature)
        self.types_input = types_input
        self.types_output = types_output

    def __call__(self, *args: TInput.args, **kwargs: TInput.kwargs) -> TOut:
        assert len(args) == len(self.types_input)
        tx_params: TxParams = {
            "to": self.contract_address,
            "data": HexStr(
                HexBytes(
                    self.selector
                    + eth_abi.encode(
                        types=self.types_input,
                        args=args,
                    )
                ).hex()
            ),
        }
        return_data = self.w3.eth.call(tx_params)
        try:
            output_data = eth_abi.decode(self.types_output, return_data)
        except DecodingError as e:
            is_missing_code_error = return_data in ACCEPTABLE_EMPTY_STRINGS and self.w3.eth.get_code(self.contract_address) in ACCEPTABLE_EMPTY_STRINGS
            if is_missing_code_error:
                msg = "Could not transact with/call contract function, is contract deployed correctly and chain synced?"
            else:
                msg = f"Could not decode contract function call to {self.signature} with return data: {str(return_data)}, output_types: {self.types_output}"
            raise BadFunctionCallOutput(msg) from e
        if len(output_data) == 1:
            return cast(TOut, output_data[0])
        return cast(TOut, output_data)


class Contract:
    def __init__(self, contract_address: ChecksumAddress | str, rpc_url: str) -> None:
        contract_address = to_checksum_address(contract_address)
        self.w3 = to_web3(rpc_url)
        self.contract_address = contract_address

        self.balanceOf = Function[[ChecksumAddress], Wei](
            w3=self.w3,
            contract_address=contract_address,
            signature="balanceOf(address)",
            types_input=["address"],
            types_output=["uint256"],
        )

        self.decimals = Function[[], int](
            w3=self.w3,
            contract_address=contract_address,
            signature="decimals()",
            types_input=[],
            types_output=["uint8"],
        )

        self.name = Function[[], str](
            w3=self.w3,
            contract_address=contract_address,
            signature="name()",
            types_input=[],
            types_output=["string"],
        )

        self.symbol = Function[[], str](
            w3=self.w3,
            contract_address=contract_address,
            signature="symbol()",
            types_input=[],
            types_output=["string"],
        )
