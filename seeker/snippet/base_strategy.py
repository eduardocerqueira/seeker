#date: 2025-11-13T17:03:06Z
#url: https://api.github.com/gists/8dde3cf7f589cb5afb9a45bd262da41a
#owner: https://api.github.com/users/wangwei334455

"""
策略基类定义
所有策略必须继承此基类
"""
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Dict, Any
from loguru import logger


class Signal(Enum):
    """交易信号枚举"""
    NONE = "none"      # 无信号
    BUY = "buy"        # 买入信号
    SELL = "sell"      # 卖出信号
    CLOSE = "close"    # 平仓信号
    HOLD = "hold"      # 持仓


class MarketMode(Enum):
    """市场模式枚举"""
    UNKNOWN = "unknown"
    RANGING = "ranging"      # 震荡
    UPTREND = "uptrend"      # 上涨
    DOWNTREND = "downtrend"  # 下跌


class BaseStrategy(ABC):
    """
    策略基类（所有策略必须继承）
    
    职责：
    1. 定义统一的策略接口
    2. 实现模式切换时的状态管理
    3. 提供指标计算和信号生成的框架
    """
    
    def __init__(self, config_manager, symbol: str = "BTCUSDm"):
        """
        初始化策略
        
        Args:
            config_manager: 配置管理器（从Redis加载参数）
            symbol: 交易品种
        """
        self.config_manager = config_manager
        self.symbol = symbol
        
        # 策略状态
        self.positions = {}  # 当前持仓
        self.orders_pending = []  # 待执行订单
        
        # 指标缓存（避免重复计算）
        self.macro_indicators = {}  # 宏观指标（ATR, BBANDS, ADX等）
        self.micro_indicators = {}  # 微观指标（LRS, TICK Density等）
    
    @abstractmethod
    def on_tick(self, tick_data: Dict[str, Any]) -> Optional[Signal]:
        """
        接收到新的TICK数据时的回调
        
        核心逻辑：
        - 实时计算微观指标（LRS、TICK Density）
        - 执行毫秒级开仓/平仓决策
        
        Args:
            tick_data: TICK数据 {time, bid, ask, last, volume, time_msc}
            
        Returns:
            Signal: 交易信号（BUY/SELL/CLOSE/HOLD/NONE）
        """
        pass
    
    @abstractmethod
    def on_kline(self, kline_data: Dict[str, Any]) -> None:
        """
        K线收盘时的回调（M1/M5）
        
        核心逻辑：
        - 更新ATR、BBANDS、ADX等宏观指标
        - 判断模式结构是否改变
        
        Args:
            kline_data: K线数据 {time, open, high, low, close, volume}
        """
        pass
    
    @abstractmethod
    def generate_signal(self) -> Optional[Signal]:
        """
        生成交易信号
        
        核心逻辑：
        - 综合宏观和微观指标
        - 根据策略规则生成信号
        
        Returns:
            Signal: 交易信号
        """
        pass
    
    @abstractmethod
    def on_mode_switch(self, new_mode: MarketMode) -> None:
        """
        模式切换时的回调（平仓逻辑）
        
        Args:
            new_mode: 新的市场模式
        """
        pass
    
    def get_config(self, mode: str, key: str, default=None):
        """
        获取配置参数（便捷方法）
        
        Args:
            mode: 配置模式（'GLOBAL', 'RANGING', 'UPTREND', 'DOWNTREND'）
            key: 参数名称
            default: 默认值
            
        Returns:
            配置值
        """
        return self.config_manager.get(mode, key, default)

