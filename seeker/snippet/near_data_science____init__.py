#date: 2026-03-02T17:37:55Z
#url: https://api.github.com/gists/3c963e0a183861e15f3b6233bff7df83
#owner: https://api.github.com/users/Quinut-McGee

"""
NEAR Data Science - A Python package for fetching, parsing, and analyzing NEAR blockchain data.

This package provides tools to interact with NEAR blockchain data using familiar pandas workflows.
"""

from .api import fetch_block, fetch_account, fetch_transactions
from .transform import parse_block_response, parse_account_response, parse_transactions_response
from .analysis import NearAccessor

__version__ = "0.1.0"
__all__ = [
    "fetch_block",
    "fetch_account", 
    "fetch_transactions",
    "parse_block_response",
    "parse_account_response",
    "parse_transactions_response",
    "NearAccessor",
]