#date: 2025-08-08T16:51:44Z
#url: https://api.github.com/gists/269ec86255cb8c6434929630db83c115
#owner: https://api.github.com/users/datavudeja

# src/__init__.py
"""
ELD - Extremely Large Data Handling Tool

This package provides a set of command-line tools for managing extremely large datasets,
including ingestion, querying, and visualization functionalities.

Modules:
    - cli: Command-line interface definitions
    - ingest: Functions for data ingestion from various file formats
    - query: Data querying capabilities
    - visualize: Data visualization tools
    - utils: Common utility functions used across different modules
"""

from .cli.cli import cli
from .ingest.ingest import ingest_data
from .query.query import detect_encoding_and_query
from .visualize.visualize import visualize_data

__version__ = '0.1.0'
__author__ = 'K45-94'
__email__ = 'hiuhukelvin@gmail.com'
__all__ = ['cli', 'ingest_data', 'detect_encoding_and_query', 'visualize_data']