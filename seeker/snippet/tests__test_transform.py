#date: 2026-03-02T17:37:55Z
#url: https://api.github.com/gists/3c963e0a183861e15f3b6233bff7df83
#owner: https://api.github.com/users/Quinut-McGee

"""
Test suite for near_data_science.transform module.

This module contains tests for data validation and transformation functions.
"""

import pandas as pd
import pytest
from near_data_science.transform import (
    parse_block_response,
    parse_account_response,
    parse_transactions_response,
    Transaction,
    Account,
    Block
)


def test_parse_transactions_response_valid():
    """Test parsing valid transaction data."""
    # Valid transaction data
    tx_data = [
        {
            "hash": "tx1",
            "signer_id": "alice.testnet",
            "receiver_id": "bob.testnet",
            "amount": 1000000000000000000000000,
            "block_timestamp": 1672531200000000000,
            "success": True
        }
    ]
    
    result = parse_transactions_response(tx_data)
    
    # Should return DataFrame with correct columns and data
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    assert result.iloc[0]["hash"] == "tx1"
    assert result.iloc[0]["signer_id"] == "alice.testnet"
    assert result.iloc[0]["receiver_id"] == "bob.testnet"
    assert result.iloc[0]["amount"] == 1000000000000000000000000
    assert result.iloc[0]["success"] == True
    
    # Should convert timestamp to datetime
    assert pd.api.types.is_datetime64_any_dtype(result["block_timestamp"])


def test_parse_transactions_response_empty():
    """Test parsing empty transaction data."""
    result = parse_transactions_response([])
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


def test_parse_transactions_response_invalid():
    """Test parsing invalid transaction data."""
    # Missing required field
    invalid_tx_data = [
        {
            "hash": "tx1",
            "signer_id": "alice.testnet",
            # Missing receiver_id
            "amount": 1000000000000000000000000,
            "block_timestamp": 1672531200000000000,
            "success": True
        }
    ]
    
    with pytest.raises(ValueError, match="Failed to parse transactions"):
        parse_transactions_response(invalid_tx_data)


def test_parse_account_response_valid():
    """Test parsing valid account data."""
    account_data = {
        "account_id": "alice.testnet",
        "state_root": "state_root_1",
        "balance": 1000000000000000000000000,
        "block_height": 1000
    }
    
    result = parse_account_response(account_data)
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    assert result.iloc[0]["account_id"] == "alice.testnet"
    assert result.iloc[0]["state_root"] == "state_root_1"
    assert result.iloc[0]["balance"] == 1000000000000000000000000
    assert result.iloc[0]["block_height"] == 1000


def test_parse_account_response_empty():
    """Test parsing empty account data."""
    result = parse_account_response({})
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


def test_parse_account_response_invalid():
    """Test parsing invalid account data."""
    # Missing required field
    invalid_account_data = {
        "account_id": "alice.testnet",
        # Missing balance
        "block_height": 1000
    }
    
    with pytest.raises(ValueError, match="Failed to parse account"):
        parse_account_response(invalid_account_data)


def test_parse_block_response_valid():
    """Test parsing valid block data."""
    block_data = {
        "height": 1,
        "hash": "block1",
        "timestamp": 1672531200000000000,
        "transactions": [
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
    
    result = parse_block_response(block_data)
    
    assert isinstance(result, pd.DataFrame)
    # Should return transaction data with block context
    assert len(result) == 1
    assert result.iloc[0]["hash"] == "tx1"
    assert result.iloc[0]["height"] == 1
    assert result.iloc[0]["hash_block"] == "block1"
    assert pd.api.types.is_datetime64_any_dtype(result["timestamp"])
    assert pd.api.types.is_datetime64_any_dtype(result["block_timestamp"])


def test_parse_block_response_empty_transactions():
    """Test parsing block data with empty transactions."""
    block_data = {
        "height": 1,
        "hash": "block1",
        "timestamp": 1672531200000000000,
        "transactions": []
    }
    
    result = parse_block_response(block_data)
    
    assert isinstance(result, pd.DataFrame)
    # Should return block data only
    assert len(result) == 1
    assert result.iloc[0]["height"] == 1
    assert result.iloc[0]["hash"] == "block1"
    assert pd.api.types.is_datetime64_any_dtype(result["timestamp"])


def test_parse_block_response_empty():
    """Test parsing empty block data."""
    result = parse_block_response({})
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


def test_parse_block_response_invalid():
    """Test parsing invalid block data."""
    # Missing required field
    invalid_block_data = {
        "height": 1,
        # Missing hash
        "timestamp": 1672531200000000000,
        "transactions": []
    }
    
    with pytest.raises(ValueError, match="Failed to parse block"):
        parse_block_response(invalid_block_data)


def test_transaction_model_validation():
    """Test Transaction model validation."""
    # Valid transaction
    valid_tx = {
        "hash": "tx1",
        "signer_id": "alice.testnet",
        "receiver_id": "bob.testnet",
        "amount": 1000000000000000000000000,
        "block_timestamp": 1672531200000000000,
        "success": True
    }
    
    transaction = Transaction(**valid_tx)
    assert transaction.hash == "tx1"
    assert transaction.signer_id == "alice.testnet"
    assert transaction.amount == 1000000000000000000000000
    
    # Invalid transaction - negative amount
    invalid_tx = valid_tx.copy()
    invalid_tx["amount"] = -1
    
    with pytest.raises(ValueError, match="Value must be positive"):
        Transaction(**invalid_tx)