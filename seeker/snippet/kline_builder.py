#date: 2025-11-13T17:03:06Z
#url: https://api.github.com/gists/8dde3cf7f589cb5afb9a45bd262da41a
#owner: https://api.github.com/users/wangwei334455

"""
K线实时构建器：负责将TICK数据聚合成指定周期的K线。

在K线收盘时，触发宏观指标计算和模式检查。
"""
from typing import Dict, Any, Optional
import numpy as np

# MT5 KLINE 数据结构简化版（只保留计算所需的字段）
# 实际项目中，应与L1/L2通信的CANDLE_DTYPE保持一致
KLINE_DTYPE = np.dtype([
    ('time', '<i8'),      # 时间戳（秒）
    ('open', '<f8'),      # 开盘价
    ('high', '<f8'),      # 最高价
    ('low', '<f8'),       # 最低价
    ('close', '<f8'),     # 收盘价
    ('volume', '<i8')     # 成交量
])


class KlineBuilder:
    """
    根据TICK数据实时构建K线。
    
    职责：
    1. 接收TICK数据，实时更新当前K线的OHLCV
    2. 检测K线收盘事件，返回闭合的K线数据
    3. 初始化新的K线
    """
    
    def __init__(self, timeframe_min: int):
        """
        初始化K线构建器
        
        Args:
            timeframe_min: K线周期（分钟），例如1表示1分钟K线
        """
        self.timeframe_sec = timeframe_min * 60
        self.current_candle: Optional[Dict[str, Any]] = None
        self.last_closed_candle: Optional[np.ndarray] = None
        self.last_candle_time: int = 0
    
    def on_tick(self, time_msc: int, price: float, volume: float = 0.0) -> Optional[np.ndarray]:
        """
        处理TICK数据，返回闭合的K线数据（如果发生收盘）。
        
        Args:
            time_msc: TICK时间戳（毫秒）
            price: TICK价格
            volume: TICK成交量（可选，默认为0）
            
        Returns:
            如果K线收盘，返回闭合的K线NumPy数组；否则返回None
        """
        current_sec = int(time_msc / 1000)
        
        # 计算当前K线应该属于哪个时间戳（K线时间的起始秒数）
        candle_start_time = current_sec - (current_sec % self.timeframe_sec)
        
        if self.current_candle is None:
            # 第一根K线或初始化
            self._initialize_new_candle(candle_start_time, price, volume)
            return None
        
        if candle_start_time > self.current_candle['time']:
            # K线收盘事件发生
            closed_candle = self._close_current_candle(candle_start_time, price)
            self.last_closed_candle = closed_candle
            self.last_candle_time = self.current_candle['time']
            
            # 初始化下一根K线
            self._initialize_new_candle(candle_start_time, price, volume)
            
            return closed_candle
        else:
            # K线持续中
            self._update_current_candle(price, volume)
            return None

    def _initialize_new_candle(self, start_time: int, price: float, volume: float):
        """
        初始化一个新的K线结构
        
        Args:
            start_time: K线开始时间（秒）
            price: 初始价格（作为开盘价）
            volume: 初始成交量
        """
        self.current_candle = {
            'time': start_time,
            'open': price,
            'high': price,
            'low': price,
            'close': price,
            'volume': volume
        }

    def _update_current_candle(self, price: float, volume: float):
        """
        更新当前K线的高低收和成交量
        
        Args:
            price: 新的TICK价格
            volume: 新的TICK成交量
        """
        if self.current_candle is not None:
            self.current_candle['high'] = max(self.current_candle['high'], price)
            self.current_candle['low'] = min(self.current_candle['low'], price)
            self.current_candle['close'] = price
            self.current_candle['volume'] += volume

    def _close_current_candle(self, next_start_time: int, current_price: float) -> np.ndarray:
        """
        关闭当前K线并返回NumPy结构体
        
        Args:
            next_start_time: 下一根K线的开始时间（秒）
            current_price: 当前TICK价格（作为收盘价）
            
        Returns:
            闭合的K线NumPy结构化数组
        """
        if self.current_candle is None:
            # 理论上不应该发生
            return np.empty(0, dtype=KLINE_DTYPE)
            
        # 确保收盘价是最后TICK的价格
        self.current_candle['close'] = current_price
        
        # 转换为NumPy结构化数组（便于传入JIT宏观指标计算）
        closed_candle_array = np.array([(
            self.current_candle['time'],
            self.current_candle['open'],
            self.current_candle['high'],
            self.current_candle['low'],
            self.current_candle['close'],
            int(self.current_candle['volume'])
        )], dtype=KLINE_DTYPE)
        
        return closed_candle_array
    
    def get_current_candle(self) -> Optional[Dict[str, Any]]:
        """
        获取当前未闭合的K线数据（用于实时监控）
        
        Returns:
            当前K线字典，如果不存在则返回None
        """
        return self.current_candle.copy() if self.current_candle else None
    
    def get_last_closed_candle(self) -> Optional[np.ndarray]:
        """
        获取最后一根闭合的K线
        
        Returns:
            最后一根闭合的K线NumPy数组，如果不存在则返回None
        """
        return self.last_closed_candle

