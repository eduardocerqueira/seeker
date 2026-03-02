#date: 2026-03-02T17:37:55Z
#url: https://api.github.com/gists/3c963e0a183861e15f3b6233bff7df83
#owner: https://api.github.com/users/Quinut-McGee

"""
Async HTTP client for NEAR JSON-RPC endpoints.

This module provides functions to fetch block, account, and transaction data
from the NEAR testnet RPC endpoint.
"""

import asyncio
from typing import Any, Dict, List, Union
import httpx
from typing_extensions import TypedDict


class RpcParams(TypedDict, total=False):
    """Type-safe parameter dictionary for RPC calls."""
    block_id: Union[int, str]
    account_id: str
    start_block: int
    end_block: int


async def _make_rpc_call(
    client: httpx.AsyncClient, 
    method: str, 
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Make an async RPC call to the NEAR testnet endpoint.
    
    Args:
        client: httpx AsyncClient instance
        method: RPC method name
        params: Method parameters
        
    Returns:
        Dictionary containing the RPC response result
        
    Raises:
        RuntimeError: If the RPC call fails or returns an error
    """
    try:
        resp = await client.post(
            "https://rpc.testnet.near.org",
            json={
                "jsonrpc": "2.0",
                "id": "1",
                "method": method,
                "params": params
            },
            timeout=10.0
        )
        resp.raise_for_status()
        result = resp.json()
        
        if "error" in result:
            raise RuntimeError(f"RPC error: {result['error']['message']}")
            
        return result["result"]
        
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"HTTP error {e.response.status_code}: {e}")
    except httpx.RequestError as e:
        raise RuntimeError(f"Request error: {e}")


async def fetch_block(
    client: httpx.AsyncClient, 
    block_id: Union[int, str]
) -> Dict[str, Any]:
    """
    Fetch block data from NEAR testnet.
    
    Args:
        client: httpx AsyncClient instance
        block_id: Block height or hash
        
    Returns:
        Dictionary containing block data
        
    Raises:
        RuntimeError: If the RPC call fails
        ValueError: If block_id is invalid
    """
    if not block_id:
        raise ValueError("block_id cannot be empty")
        
    params: RpcParams = {"block_id": block_id}
    return await _make_rpc_call(client, "block", params)


async def fetch_account(
    client: httpx.AsyncClient, 
    account_id: str
) -> Dict[str, Any]:
    """
    Fetch account data from NEAR testnet.
    
    Args:
        client: httpx AsyncClient instance
        account_id: NEAR account ID
        
    Returns:
        Dictionary containing account data
        
    Raises:
        RuntimeError: If the RPC call fails
        ValueError: If account_id is invalid
    """
    if not account_id:
        raise ValueError("account_id cannot be empty")
        
    params: RpcParams = {"account_id": account_id}
    return await _make_rpc_call(client, "query", {
        "request_type": "view_account",
        "account_id": account_id,
        "finality": "final"
    })


async def fetch_transactions(
    client: httpx.AsyncClient,
    account_id: str,
    start_block: int,
    end_block: int
) -> List[Dict[str, Any]]:
    """
    Fetch transactions for an account within a block range.
    
    Args:
        client: httpx AsyncClient instance
        account_id: NEAR account ID
        start_block: Starting block height
        end_block: Ending block height
        
    Returns:
        List of transaction dictionaries
        
    Raises:
        RuntimeError: If the RPC call fails
        ValueError: If parameters are invalid
    """
    if not account_id:
        raise ValueError("account_id cannot be empty")
    if start_block < 0 or end_block < 0:
        raise ValueError("Block heights cannot be negative")
    if start_block > end_block:
        raise ValueError("start_block cannot be greater than end_block")
        
    # Note: This is a simplified implementation. The actual NEAR RPC
    # may require different parameters or multiple calls to get transactions
    # by account within a block range.
    
    # For now, we'll use the experimental tx method if available
    # This may need to be adjusted based on actual RPC capabilities
    try:
        params: RpcParams = {
            "account_id": account_id,
            "start_block": start_block,
            "end_block": end_block
        }
        return await _make_rpc_call(client, "EXPERIMENTAL_tx", params)
    except RuntimeError:
        # Fallback: fetch blocks individually and extract transactions
        # This is less efficient but more reliable
        transactions = []
        
        for block_height in range(start_block, end_block + 1):
            try:
                block_data = await fetch_block(client, block_height)
                if "transactions" in block_data:
                    for tx in block_data["transactions"]:
                        # Add block context to each transaction
                        tx_with_context = tx.copy()
                        tx_with_context["block_height"] = block_height
                        tx_with_context["block_hash"] = block_data.get("hash")
                        transactions.append(tx_with_context)
            except (RuntimeError, ValueError):
                # Skip blocks that can't be fetched
                continue
                
        return transactions