#date: 2026-03-02T17:37:55Z
#url: https://api.github.com/gists/3c963e0a183861e15f3b6233bff7df83
#owner: https://api.github.com/users/Quinut-McGee

"""
Data validation and transformation for NEAR blockchain data.

This module provides pydantic models and DataFrame parsing functions
to convert raw JSON responses from the NEAR RPC into structured pandas DataFrames.
"""

from typing import List, Optional
import pandas as pd
from pydantic import BaseModel, Field, field_validator


class Transaction(BaseModel):
    """Pydantic model for NEAR transaction data."""
    
    hash: str = Field(..., description="Transaction hash")
    signer_id: str = Field(..., description="Account ID of transaction signer")
    receiver_id: str = Field(..., description="Account ID of transaction receiver")
    amount: int = Field(..., description="Transaction amount in yoctoNEAR")
    block_timestamp: int = Field(..., description="Block timestamp in Unix nanoseconds")
    success: bool = Field(..., description="Whether transaction succeeded")
    
    @field_validator('hash', 'signer_id', 'receiver_id')
    @classmethod
    def validate_string_fields(cls, v):
        """Validate that string fields are not empty."""
        if not v or not v.strip():
            raise ValueError("String field cannot be empty")
        return v.strip()
    
    @field_validator('amount', 'block_timestamp')
    @classmethod
    def validate_positive_int(cls, v):
        """Validate that integer fields are positive."""
        if v < 0:
            raise ValueError("Value must be positive")
        return v


class Account(BaseModel):
    """Pydantic model for NEAR account data."""
    
    account_id: str = Field(..., description="NEAR account ID")
    state_root: str = Field(..., description="State root hash")
    balance: int = Field(..., description="Account balance in yoctoNEAR")
    block_height: int = Field(..., description="Block height when account was queried")
    
    @field_validator('account_id', 'state_root')
    @classmethod
    def validate_string_fields(cls, v):
        """Validate that string fields are not empty."""
        if not v or not v.strip():
            raise ValueError("String field cannot be empty")
        return v.strip()
    
    @field_validator('balance', 'block_height')
    @classmethod
    def validate_positive_int(cls, v):
        """Validate that integer fields are positive."""
        if v < 0:
            raise ValueError("Value must be positive")
        return v


class Block(BaseModel):
    """Pydantic model for NEAR block data."""
    
    height: int = Field(..., description="Block height")
    hash: str = Field(..., description="Block hash")
    timestamp: int = Field(..., description="Block timestamp in Unix nanoseconds")
    transactions: List[Transaction] = Field(default_factory=list, description="List of transactions in block")
    
    @field_validator('hash')
    @classmethod
    def validate_hash(cls, v):
        """Validate that hash is not empty."""
        if not v or not v.strip():
            raise ValueError("Hash cannot be empty")
        return v.strip()
    
    @field_validator('height', 'timestamp')
    @classmethod
    def validate_positive_int(cls, v):
        """Validate that integer fields are positive."""
        if v < 0:
            raise ValueError("Value must be positive")
        return v


def parse_transactions_response(data: List[dict]) -> pd.DataFrame:
    """
    Parse raw transaction data into a pandas DataFrame.
    
    Args:
        data: List of raw transaction dictionaries from RPC
        
    Returns:
        pandas DataFrame with structured transaction data
        
    Raises:
        ValueError: If data validation fails
    """
    if not data:
        return pd.DataFrame()
    
    try:
        # Validate and convert each transaction
        transactions = []
        for tx_data in data:
            tx = Transaction(**tx_data)
            transactions.append(tx.model_dump())
        
        # Create DataFrame and convert timestamp
        df = pd.DataFrame(transactions)
        if not df.empty and 'block_timestamp' in df.columns:
            df['block_timestamp'] = pd.to_datetime(df['block_timestamp'], unit='ns')
        
        return df
        
    except Exception as e:
        raise ValueError(f"Failed to parse transactions: {e}")


def parse_account_response(data: dict) -> pd.DataFrame:
    """
    Parse raw account data into a pandas DataFrame.
    
    Args:
        data: Raw account dictionary from RPC
        
    Returns:
        pandas DataFrame with structured account data
        
    Raises:
        ValueError: If data validation fails
    """
    if not data:
        return pd.DataFrame()
    
    try:
        # Validate and convert account data
        account = Account(**data)
        account_dict = account.model_dump()
        
        # Create DataFrame with single row
        df = pd.DataFrame([account_dict])
        
        return df
        
    except Exception as e:
        raise ValueError(f"Failed to parse account: {e}")


def parse_block_response(data: dict) -> pd.DataFrame:
    """
    Parse raw block data into a pandas DataFrame.
    
    Args:
        data: Raw block dictionary from RPC
        
    Returns:
        pandas DataFrame with structured block data
        
    Raises:
        ValueError: If data validation fails
    """
    if not data:
        return pd.DataFrame()
    
    try:
        # Validate and convert block data
        block = Block(**data)
        block_dict = block.model_dump()
        
        # Extract transactions separately for normalization
        transactions = block_dict.pop('transactions', [])
        
        # Create block DataFrame
        block_df = pd.DataFrame([block_dict])
        
        # Convert timestamp
        if 'timestamp' in block_df.columns:
            block_df['timestamp'] = pd.to_datetime(block_df['timestamp'], unit='ns')
        
        # If there are transactions, parse them and join with block data
        if transactions:
            tx_df = parse_transactions_response(transactions)
            # Add block context to each transaction
            for col in block_df.columns:
                if col != 'transactions':
                    tx_df[col] = block_df[col].iloc[0]
            return tx_df
        
        return block_df
        
    except Exception as e:
        raise ValueError(f"Failed to parse block: {e}")


def _flatten_nested_data(data: dict) -> dict:
    """
    Flatten nested dictionary structures for better DataFrame compatibility.
    
    Args:
        data: Nested dictionary
        
    Returns:
        Flattened dictionary
    """
    flattened = {}
    for key, value in data.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                flattened[f"{key}_{subkey}"] = subvalue
        else:
            flattened[key] = value
    return flattened