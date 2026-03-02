#date: 2026-03-02T17:37:55Z
#url: https://api.github.com/gists/3c963e0a183861e15f3b6233bff7df83
#owner: https://api.github.com/users/Quinut-McGee

"""
Test suite for near_data_science.api module.

This module contains tests for the async RPC client functions.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, patch
from near_data_science.api import fetch_block, fetch_account, fetch_transactions


def test_fetch_block_valid_input():
    """Test fetch_block with valid block ID."""
    # Mock the async client
    mock_client = AsyncMock()
    mock_response = {
        "jsonrpc": "2.0",
        "id": "1",
        "result": {
            "height": 1,
            "hash": "block1",
            "timestamp": 1672531200000000000,
            "transactions": []
        }
    }
    mock_client.post.return_value = AsyncMock(json=AsyncMock(return_value=mock_response))
    
    # Test with valid block ID
    result = asyncio.run(fetch_block(mock_client, 1))
    assert result['height'] == 1
    assert result['hash'] == 'block1'
    assert result['timestamp'] == 1672531200000000000
    
    # Verify the client was called with correct parameters
    mock_client.post.assert_called_once()
    call_args = mock_client.post.call_args
    assert call_args[0][0] == "https://rpc.testnet.near.org"
    assert call_args[1]['json']['method'] == "block"
    assert call_args[1]['json']['params']['block_id'] == 1


def test_fetch_block_invalid_input():
    """Test fetch_block with invalid block ID."""
    mock_client = AsyncMock()
    
    # Test with empty block ID
    with pytest.raises(ValueError, match="block_id cannot be empty"):
        asyncio.run(fetch_block(mock_client, ""))
    
    # Test with None block ID
    with pytest.raises(ValueError, match="block_id cannot be empty"):
        asyncio.run(fetch_block(mock_client, None))


def test_fetch_account_valid_input():
    """Test fetch_account with valid account ID."""
    # Mock the async client
    mock_client = AsyncMock()
    mock_response = {
        "jsonrpc": "2.0",
        "id": "1",
        "result": {
            "account_id": "alice.testnet",
            "state_root": "state_root_1",
            "balance": 1000000000000000000000000,
            "block_height": 1000
        }
    }
    mock_client.post.return_value = AsyncMock(json=AsyncMock(return_value=mock_response))
    
    # Test with valid account ID
    result = asyncio.run(fetch_account(mock_client, "alice.testnet"))
    assert result['account_id'] == "alice.testnet"
    assert result['state_root'] == "state_root_1"
    assert result['balance'] == 1000000000000000000000000
    assert result['block_height'] == 1000
    
    # Verify the client was called with correct parameters
    mock_client.post.assert_called_once()
    call_args = mock_client.post.call_args
    assert call_args[0][0] == "https://rpc.testnet.near.org"
    assert call_args[1]['json']['method'] == "query"
    assert call_args[1]['json']['params']['request_type'] == "view_account"
    assert call_args[1]['json']['params']['account_id'] == "alice.testnet"


def test_fetch_account_invalid_input():
    """Test fetch_account with invalid account ID."""
    mock_client = AsyncMock()
    
    # Test with empty account ID
    with pytest.raises(ValueError, match="account_id cannot be empty"):
        asyncio.run(fetch_account(mock_client, ""))
    
    # Test with None account ID
    with pytest.raises(ValueError, match="account_id cannot be empty"):
        asyncio.run(fetch_account(mock_client, None))


def test_fetch_transactions_valid_input():
    """Test fetch_transactions with valid parameters."""
    # Mock the async client
    mock_client = AsyncMock()
    mock_response = {
        "jsonrpc": "2.0",
        "id": "1",
        "result": [
            {
                "hash": "tx1",
                "signer_id": "alice.testnet",
                "receiver_id": "bob.testnet",
                "amount": 1000000000000000000000000,
                "block_timestamp": 1672531200000000000,
                "success": True
            }
        ]
    }
    mock_client.post.return_value = AsyncMock(json=AsyncMock(return_value=mock_response))
    
    # Test with valid parameters
    result = asyncio.run(fetch_transactions(mock_client, "alice.testnet", 1, 10))
    assert len(result) == 1
    assert result[0]['hash'] == "tx1"
    assert result[0]['signer_id'] == "alice.testnet"
    assert result[0]['receiver_id'] == "bob.testnet"
    assert result[0]['amount'] == 1000000000000000000000000
    assert result[0]['block_timestamp'] == 1672531200000000000
    assert result[0]['success'] == True
    
    # Verify the client was called with correct parameters
    mock_client.post.assert_called_once()
    call_args = mock_client.post.call_args
    assert call_args[0][0] == "https://rpc.testnet.near.org"
    assert call_args[1]['json']['method'] == "EXPERIMENTAL_tx"
    assert call_args[1]['json']['params']['account_id'] == "alice.testnet"
    assert call_args[1]['json']['params']['start_block'] == 1
    assert call_args[1]['json']['params']['end_block'] == 10


def test_fetch_transactions_invalid_input():
    """Test fetch_transactions with invalid parameters."""
    mock_client = AsyncMock()
    
    # Test with empty account ID
    with pytest.raises(ValueError, match="account_id cannot be empty"):
        asyncio.run(fetch_transactions(mock_client, "", 1, 10))
    
    # Test with None account ID
    with pytest.raises(ValueError, match="account_id cannot be empty"):
        asyncio.run(fetch_transactions(mock_client, None, 1, 10))
    
    # Test with negative block heights
    with pytest.raises(ValueError, match="Block heights cannot be negative"):
        asyncio.run(fetch_transactions(mock_client, "alice.testnet", -1, 10))
    
    # Test with start_block > end_block
    with pytest.raises(ValueError, match="start_block cannot be greater than end_block"):
        asyncio.run(fetch_transactions(mock_client, "alice.testnet", 10, 1))