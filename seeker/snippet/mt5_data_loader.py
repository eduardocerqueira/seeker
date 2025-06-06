#date: 2025-06-06T17:10:15Z
#url: https://api.github.com/gists/7f2ae4d9425288ff3a390362a51dc9c8
#owner: https://api.github.com/users/vr856

# MT5 Data Loader for DQN Trading Strategy
# Loads historical data from MT5 JSON format

import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class MT5DataLoader:
    """Load and process MT5 JSON data for DQN training"""
    
    def __init__(self, json_file: str):
        self.json_file = json_file
        self.data = None
        self.df = None
        
    def load_data(self) -> bool:
        """Load JSON data from file"""
        try:
            with open(self.json_file, 'r') as f:
                self.data = json.load(f)
            
            logger.info(f"Successfully loaded {len(self.data)} records from {self.json_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading MT5 data: {e}")
            return False
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert JSON data to pandas DataFrame"""
        if self.data is None:
            self.load_data()
        
        # Convert to DataFrame
        df = pd.DataFrame(self.data)
        
        # Convert timestamp to datetime
        df['datetime'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('datetime', inplace=True)
        
        # Rename columns to standard OHLCV format
        df = df.rename(columns={
            'open': 'open',
            'high': 'high', 
            'low': 'low',
            'close': 'close',
            'tick_volume': 'volume'
        })
        
        # Ensure numeric types
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove any NaN rows
        df = df.dropna()
        
        # Sort by datetime
        df = df.sort_index()
        
        self.df = df
        logger.info(f"Converted to DataFrame: {len(df)} bars from {df.index[0]} to {df.index[-1]}")
        
        return df

# Project Configuration (pyproject.toml equivalent)
PROJECT_CONFIG = '''
[build-system]
requires = ["setuptools>=45", "wheel", "setuptools_scm[toml]>=6.2"]
build-backend = "setuptools.build_meta"

[project]
name = "dqn-trading-strategy"
version = "1.0.0"
description = "Deep Q-Network trading strategy for 1-minute cryptocurrency trading"
authors = [
    {name = "DQN Trader", email = "trader@example.com"}
]
license = {text = "MIT"}
readme = "README.md"
requires-python = ">=3.9"

dependencies = [
    "matplotlib>=3.10.3",
    "numpy>=2.2.6", 
    "pandas>=2.3.0",
    "ta>=0.11.0",
    "torch>=2.0.0",
    "torchvision>=0.15.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "black>=22.0",
    "flake8>=4.0",
    "mypy>=0.900",
]

[tool.setuptools]
packages = ["dqn_strategy"]

[tool.black]
line-length = 88
target-version = ['py39']

[tool.mypy]
python_version = "3.9"
warn_return_any = true
warn_unused_configs = true
'''

# Strategy Documentation Header
STRATEGY_DOCS = '''
# DQN 1-Minute Trading Strategy

## Overview
Advanced Deep Q-Network implementation for cryptocurrency trading with:
- Real-time MT5 data integration
- Technical indicator-based state representation
- Experience replay buffer for stable learning
- Live inference capabilities

## Key Features
✅ PyTorch-based DQN implementation
✅ MT5 JSON data loader
✅ Technical analysis indicators
✅ Position management system
✅ Performance metrics tracking
✅ Model persistence and loading

## Complete File Structure
- DQN-winning-strat.py (47KB) - Main strategy implementation
- mt5_data_loader.py - MT5 data integration
- dqn_inference_example.py - Live trading inference
- README_Strategy.md - Comprehensive documentation
- Trained model checkpoints and utilities

## Usage
```python
from DQN_winning_strat import DQNTrainer, DQNConfig

config = DQNConfig()
trainer = DQNTrainer(config)
trainer.train(episodes=1000, start_date=..., end_date=...)
```
'''

if __name__ == "__main__":
    print("MT5 Data Loader for DQN Trading Strategy")
    print("Load historical BTCUSD data for training and backtesting")