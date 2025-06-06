#date: 2025-06-06T17:03:30Z
#url: https://api.github.com/gists/e96152127b2b0a13595282ff7efa7815
#owner: https://api.github.com/users/vr856

# DQN Trading Strategy - High Net Low Drawdown (MT5 Only)
# AI-powered trading bot using Deep Q-Network with PyTorch
# Uses MT5 JSON data exclusively

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import ta  # Using ta library instead of talib for consistency
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt
import random
from collections import deque
import pickle
import json
import hashlib
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set device to CPU
device = torch.device("cpu")
logger.info(f"Using device: {device}")

class Action(Enum):
    HOLD = 0
    BUY = 1
    SELL = 2

class PositionType(Enum):
    FLAT = 0
    LONG = 1
    SHORT = 2

@dataclass
class DQNConfig:
    """DQN Configuration"""
    # Environment settings
    symbol: str = "BTCUSD"
    mt5_file: str = "Data/BTCUSD1.json"
    initial_balance: float = 10000.0
    position_size: float = 0.01
    max_steps: int = 1000
    lookback_window: int = 50
    
    # Technical indicator parameters (matching original strategy)
    fast_ma_period: int = 5
    slow_ma_period: int = 15
    cci_period: int = 24
    atr_period: int = 26
    atr_level: float = 246.0
    
    # DQN hyperparameters
    state_size: int = 20  # Will be calculated based on features
    action_size: int = 3  # HOLD, BUY, SELL
    learning_rate: float = 0.001
    gamma: float = 0.95  # Discount factor
    epsilon: float = 1.0  # Exploration rate
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    batch_size: int = 32
    memory_size: int = 10000
    target_update_freq: int = 100
    
    # Trading parameters
    transaction_cost: float = 0.001  # 0.1% transaction cost
    max_position_time: int = 100  # Max bars to hold position

# [Note: This is a preview of the main file. Full code contains 1141 lines including:]
# - Complete DQN Network implementation with PyTorch
# - Trading environment with MT5 data integration
# - Experience replay buffer
# - DQN Agent with epsilon-greedy strategy
# - Training and testing framework
# - Live inference capabilities
# - Model persistence and metadata tracking

class DQNNetwork(nn.Module):
    """Deep Q-Network using PyTorch"""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super(DQNNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(hidden_size, 64)
        self.dropout3 = nn.Dropout(0.1)
        self.fc4 = nn.Linear(64, 64)
        self.fc5 = nn.Linear(64, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        x = F.relu(self.fc4(x))
        x = self.dropout3(x)
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

# The complete implementation includes:
# - TradingEnvironment class with MT5 data loading
# - Technical indicator calculations
# - DQN Agent with experience replay
# - Training loop with performance metrics
# - Live inference system
# - Model serialization and loading
# - Comprehensive logging and monitoring

if __name__ == "__main__":
    print("DQN Trading Strategy - Ready for training and inference")
    print("Run with appropriate configuration for your trading setup")