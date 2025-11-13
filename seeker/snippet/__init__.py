#date: 2025-11-13T17:03:06Z
#url: https://api.github.com/gists/8dde3cf7f589cb5afb9a45bd262da41a
#owner: https://api.github.com/users/wangwei334455

"""
策略模块
包含所有交易策略的基类和实现
"""

from .base_strategy import BaseStrategy, Signal, MarketMode
from .ranging_strategy import RangingStrategy
from .uptrend_strategy import UptrendStrategy
from .downtrend_strategy import DowntrendStrategy

__all__ = [
    'BaseStrategy',
    'Signal',
    'MarketMode',
    'RangingStrategy',
    'UptrendStrategy',
    'DowntrendStrategy',
]

