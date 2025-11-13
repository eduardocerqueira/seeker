#date: 2025-11-13T17:03:06Z
#url: https://api.github.com/gists/8dde3cf7f589cb5afb9a45bd262da41a
#owner: https://api.github.com/users/wangwei334455

"""
K线实时构建器（增强版）：支持多时间周期

从TICK数据实时构建和维护多时间周期K线。

【核心功能】
1. 多时间周期支持（M1, M5, H1等）
2. 原子更新：实时处理TICK数据，原子性更新OHLCV
3. 周期闭合检测：精确检测K线周期闭合，触发宏观指标计算
4. 历史加载集成：接收并初始化历史K线数据
"""
import time
import math
from typing import Dict, Any, List, Optional
from loguru import logger
import numpy as np

# 时间周期常量（秒）
TIMEFRAME_TO_SECONDS = {
    "M1": 60,
    "M5": 300,
    "H1": 3600,
    "H4": 14400,
    "D1": 86400
}

# MT5 KLINE 数据结构
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
    K线实时构建器：从TICK数据实时构建和维护多时间周期K线
    
    【职责】
    1. 接收TICK数据，实时更新当前K线的OHLCV
    2. 检测K线收盘事件，返回闭合的K线数据
    3. 初始化新的K线
    4. 支持多时间周期同时构建
    """
    
    def __init__(self, symbol: str, active_timeframes: List[str] = ["M1"]):
        """
        初始化K线构建器
        
        Args:
            symbol: 交易品种
            active_timeframes: 激活的时间周期列表（例如：["M1", "M5", "H1"]）
        """
        self.symbol = symbol
        self.active_timeframes = active_timeframes
        
        # 存储K线状态的字典: {timeframe: {'current': kline_dict, 'history': list}}
        self.kline_states: Dict[str, Dict[str, Any]] = {}
        
        # 用于通知MacroIndicators更新的标志
        self.closed_bars_ready: Dict[str, Optional[Dict]] = {tf: None for tf in active_timeframes}
        
        self._initialize_states()
        logger.info(f"K线构建器 ({symbol}) 初始化成功，激活周期: {active_timeframes}")
    
    def _initialize_states(self):
        """初始化K线状态结构"""
        for tf in self.active_timeframes:
            self.kline_states[tf] = {
                'current': self._create_empty_kline(),
                'history': [],  # 存储已闭合的历史K线
                'period_seconds': TIMEFRAME_TO_SECONDS.get(tf, 60),
                'last_tick_time': 0.0  # 上一个处理的TICK时间（用于检测时间倒流）
            }
    
    def _create_empty_kline(self) -> Dict[str, Any]:
        """创建一个空的K线字典"""
        return {
            'time': 0,      # K线起始时间（秒级时间戳）
            'open': 0.0,
            'high': -float('inf'),
            'low': float('inf'),
            'close': 0.0,
            'volume': 0,
            'is_closed': False
        }
    
    # --- 1. 历史数据加载 ---
    
    def load_history(self, timeframe: str, historical_klines: List[Dict]):
        """
        加载历史K线到指定周期，用于MacroContext初始化
        
        Args:
            timeframe: 时间周期（"M1", "M5"等）
            historical_klines: Redis获取的历史K线列表
        """
        if timeframe not in self.kline_states:
            logger.warning(f"周期 {timeframe} 未激活，无法加载历史数据")
            return
        
        if historical_klines:
            state = self.kline_states[timeframe]
            
            # 导入所有历史数据
            state['history'].extend(historical_klines)
            
            # 尝试从最后一个历史K线恢复当前K线（通常会忽略当前K线，等待新TICK）
            # 我们将最后一个K线视为已闭合，等待新TICK开启新K线
            
            logger.info(f"✓ 周期 {timeframe} 成功加载 {len(historical_klines)} 根历史K线")
    
    # --- 2. 实时TICK处理核心 ---
    
    def process_tick(self, tick: Dict[str, Any]) -> Dict[str, Optional[np.ndarray]]:
        """
        处理单条TICK数据，更新所有激活周期K线
        
        Args:
            tick: 从Redis Stream收到的TICK数据（包含bid, ask, last, time_msc, seq等）
            
        Returns:
            Dict[str, Optional[np.ndarray]]: 各周期闭合的K线字典，如果未闭合则为None
        """
        tick_time_ms = tick.get('time_msc', tick.get('time', 0) * 1000)
        tick_time_s = int(tick_time_ms / 1000)
        last_price = tick.get('last')  # 默认使用last price
        
        if last_price is None or last_price <= 0:
            # 如果没有last price（或异常），使用(bid+ask)/2
            if tick.get('bid') and tick.get('ask'):
                last_price = (tick['bid'] + tick['ask']) / 2.0
            else:
                logger.warning(f"TICK数据无效，跳过处理 (Seq: {tick.get('seq')})")
                return {tf: None for tf in self.active_timeframes}
        
        closed_bars = {}
        
        for tf, state in self.kline_states.items():
            # A. 检查时间倒流（虽然Seq已经检查，但以防万一）
            if tick_time_s < state['last_tick_time']:
                logger.error(f"时间倒流检测！TICK时间 {tick_time_s} < 上次时间 {state['last_tick_time']}。跳过。")
                closed_bars[tf] = None
                continue
            state['last_tick_time'] = tick_time_s
            
            # B. 确定当前K线的起始时间
            period_sec = state['period_seconds']
            # 计算当前K线应有的起始时间（以整数倍周期秒计算）
            bar_start_time = math.floor(tick_time_s / period_sec) * period_sec
            
            current_bar = state['current']
            
            # C. K线闭合检测（最核心逻辑）
            if current_bar['time'] == 0:
                # 首次接收TICK或历史加载后，初始化当前K线
                self._initialize_current_bar(current_bar, bar_start_time, last_price)
                closed_bars[tf] = None
                
            elif bar_start_time > current_bar['time']:
                # 周期闭合！
                closed_bar = self._close_bar(current_bar, last_price)
                
                # 转换为NumPy数组（便于传入JIT宏观指标计算）
                closed_bar_array = self._dict_to_numpy(closed_bar)
                
                # 记录已闭合K线，等待下游消费
                self.closed_bars_ready[tf] = closed_bar
                state['history'].append(closed_bar)
                closed_bars[tf] = closed_bar_array
                
                # 开启新的K线
                self._initialize_current_bar(current_bar, bar_start_time, last_price)
                
            # D. K线更新（如果K线未闭合，且TICK属于当前周期）
            elif bar_start_time == current_bar['time']:
                self._update_current_bar(current_bar, last_price, tick.get('volume', 0))
                closed_bars[tf] = None
            
            # E. 延迟警告（如果TICK时间远远超过了当前bar_start_time）
            # 这是一个次要检查，通常由StrategyFSM的延迟监控处理
            # if tick_time_s > bar_start_time + period_sec:
            #     logger.warning(f"TICK延迟过大，可能丢失数据：{tick_time_s - bar_start_time}s")
        
        return closed_bars
    
    def _initialize_current_bar(self, bar: Dict, start_time: int, price: float):
        """开启一个新的K线"""
        bar.update({
            'time': start_time,
            'open': price,
            'high': price,
            'low': price,
            'close': price,
            'volume': 0,
            'is_closed': False
        })
    
    def _update_current_bar(self, bar: Dict, price: float, volume_increment: int):
        """更新当前K线的OHLCV"""
        bar['high'] = max(bar['high'], price)
        bar['low'] = min(bar['low'], price)
        bar['close'] = price  # 最新价格即为当前的收盘价
        bar['volume'] += volume_increment
    
    def _close_bar(self, bar: Dict, last_price: float) -> Dict:
        """K线闭合处理，最终确定收盘价和状态"""
        closed_bar = bar.copy()
        # 确保收盘价使用闭合前的最后一个价格（即传入的last_price）
        closed_bar['close'] = last_price
        closed_bar['is_closed'] = True
        return closed_bar
    
    def _dict_to_numpy(self, kline_dict: Dict) -> np.ndarray:
        """将K线字典转换为NumPy结构化数组"""
        return np.array([(
            kline_dict['time'],
            kline_dict['open'],
            kline_dict['high'],
            kline_dict['low'],
            kline_dict['close'],
            int(kline_dict['volume'])
        )], dtype=KLINE_DTYPE)
    
    # --- 3. 兼容旧接口（单周期）---
    
    def on_tick(self, time_msc: int, price: float, volume: float = 0.0) -> Optional[np.ndarray]:
        """
        处理TICK数据（兼容旧接口，仅处理M1周期）
        
        Args:
            time_msc: TICK时间戳（毫秒）
            price: TICK价格
            volume: TICK成交量（可选，默认为0）
            
        Returns:
            如果K线收盘，返回闭合的K线NumPy数组；否则返回None
        """
        tick = {
            'time_msc': time_msc,
            'time': int(time_msc / 1000),
            'last': price,
            'volume': int(volume)
        }
        
        closed_bars = self.process_tick(tick)
        return closed_bars.get('M1', None)
    
    # --- 4. 供MacroIndicators调用的接口 ---
    
    def get_last_closed_bar(self, timeframe: str) -> Optional[Dict]:
        """获取最近闭合的K线，并清除标志"""
        if timeframe not in self.closed_bars_ready:
            return None
        
        closed_bar = self.closed_bars_ready[timeframe]
        
        # 消费后清空，避免重复计算
        self.closed_bars_ready[timeframe] = None
        
        return closed_bar
    
    def get_history_for_calc(self, timeframe: str, lookback: int) -> List[Dict]:
        """
        获取用于指标计算的历史数据（历史 + 当前Bar）
        
        Args:
            timeframe: 时间周期
            lookback: 需要的历史K线数量
            
        Returns:
            K线数据列表
        """
        if timeframe not in self.kline_states:
            return []
        
        state = self.kline_states[timeframe]
        history = state['history']
        current = state['current']
        
        # 组合历史和当前K线（用于最新的价格更新，例如计算最新的RSI/BBANDS）
        klines = history[-lookback:] + [current]
        
        return klines
    
    def is_bar_closed(self, timeframe: str) -> bool:
        """检查是否有新的K线闭合"""
        return self.closed_bars_ready.get(timeframe) is not None
    
    def get_current_candle(self, timeframe: str = "M1") -> Optional[Dict[str, Any]]:
        """
        获取当前未闭合的K线数据（用于实时监控）
        
        Args:
            timeframe: 时间周期（默认M1）
            
        Returns:
            当前K线字典，如果不存在则返回None
        """
        if timeframe not in self.kline_states:
            return None
        
        current = self.kline_states[timeframe]['current']
        return current.copy() if current['time'] > 0 else None

