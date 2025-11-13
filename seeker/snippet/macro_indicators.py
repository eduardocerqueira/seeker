#date: 2025-11-13T17:03:06Z
#url: https://api.github.com/gists/8dde3cf7f589cb5afb9a45bd262da41a
#owner: https://api.github.com/users/wangwei334455

"""
宏观指标计算模块：包含基于K线的高速JIT编译函数。

- ATR (平均真实波幅)
- BBANDS (布林带)
- ADX (平均趋向指数)
- RSI (相对强弱指标)
"""
import numpy as np
from numba import njit, float64, int64
from collections import deque
from typing import Deque, Dict, Any, Union, Tuple

# 导入KLINE_DTYPE（从kline_builder模块）
from ..kline_builder import KLINE_DTYPE

# --- 定义JIT宏观指标需要的K线字段索引 ---
# 必须与KlineBuilder或L1 Pusher的KLINE_DTYPE顺序一致
TIME, OPEN, HIGH, LOW, CLOSE, VOLUME = 0, 1, 2, 3, 4, 5


# --- 1. ATR (平均真实波幅) JIT 函数 ---

@njit(float64[:](float64[:], float64[:], float64[:], int64))
def calculate_atr_jit(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """
    Numba JIT编译的ATR (Average True Range) 计算。
    
    使用平滑方法（SMA或EMA类似）进行计算。
    
    Args:
        high: 最高价数组
        low: 最低价数组
        close: 收盘价数组
        period: ATR计算周期
        
    Returns:
        ATR值数组
    """
    N = len(close)
    if N < period:
        return np.array([0.0])  # 无法计算

    TR = np.zeros(N, dtype=np.float64)
    ATR = np.zeros(N, dtype=np.float64)

    for i in range(1, N):
        # 计算真实波幅 (True Range, TR)
        tr1 = high[i] - low[i]
        tr2 = np.abs(high[i] - close[i-1])
        tr3 = np.abs(low[i] - close[i-1])
        TR[i] = max(tr1, max(tr2, tr3))

    # 计算初始ATR（使用SMA）
    if period > 1:
        initial_sum = 0.0
        for i in range(1, period):
            initial_sum += TR[i]
        ATR[period - 1] = initial_sum / (period - 1)
    else:
        ATR[0] = TR[1] if N > 1 else 0.0

    # 后续使用EMA-like平滑方法
    for i in range(period, N):
        ATR[i] = (ATR[i-1] * (period - 1) + TR[i]) / period
        
    return ATR


# --- 2. BBANDS (布林带) JIT 函数 ---

@njit
def calculate_bbands_jit(close: np.ndarray, period: int, std_dev_mult: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Numba JIT编译的布林带计算。
    
    Returns:
        (中轨, 上轨, 下轨)
        
    Args:
        close: 收盘价数组
        period: 计算周期
        std_dev_mult: 标准差倍数
        
    Returns:
        (中轨数组, 上轨数组, 下轨数组)
    """
    N = len(close)
    if N < period:
        empty = np.array([0.0])
        return empty, empty, empty

    MA = np.zeros(N, dtype=np.float64)
    SD = np.zeros(N, dtype=np.float64)

    # 1. 计算SMA（中轨）和标准差（SD）
    for i in range(period - 1, N):
        # 手动计算均值和标准差（JIT优化）
        window_sum = 0.0
        for j in range(i - period + 1, i + 1):
            window_sum += close[j]
        
        mean_val = window_sum / period
        MA[i] = mean_val
        
        # 计算方差
        variance_sum = 0.0
        for j in range(i - period + 1, i + 1):
            diff = close[j] - mean_val
            variance_sum += diff * diff
        
        variance = variance_sum / period
        SD[i] = np.sqrt(variance)

    # 2. 计算上轨和下轨
    UpperBand = MA + SD * std_dev_mult
    LowerBand = MA - SD * std_dev_mult
    
    return MA, UpperBand, LowerBand


# --- 辅助函数：TR 计算 (供 ATR 和 ADX 内部使用) ---

@njit(float64[:](float64[:], float64[:], float64[:]))
def _calculate_tr_array(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """
    计算 True Range (TR) 数组
    
    Args:
        high: 最高价数组
        low: 最低价数组
        close: 收盘价数组
        
    Returns:
        TR数组
    """
    N = len(close)
    TR = np.zeros(N, dtype=np.float64)
    if N == 0:
        return TR
    
    # 第一根K线的TR无法依赖前一根K线，这里只关注波幅
    TR[0] = high[0] - low[0]
    
    for i in range(1, N):
        tr1 = high[i] - low[i]
        tr2 = np.abs(high[i] - close[i-1])
        tr3 = np.abs(low[i] - close[i-1])
        TR[i] = max(tr1, max(tr2, tr3))
    
    return TR


# --- 3. ADX (平均趋向指数) JIT 函数 ---

@njit(float64[:](float64[:], float64[:], float64[:], int64))
def calculate_adx_jit(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """
    Numba JIT编译的ADX (Average Directional Index) 计算
    
    Args:
        high: 最高价数组
        low: 最低价数组
        close: 收盘价数组
        period: 计算周期
        
    Returns:
        ADX值数组
    """
    N = len(close)
    if N < 2 * period:
        # 至少需要2*period根K线才能得到第一个可靠的ADX值
        return np.zeros(N, dtype=np.float64)
    
    # Step 1: 计算TR, DM+, DM-
    TR = _calculate_tr_array(high, low, close)
    DM_plus = np.zeros(N, dtype=np.float64)
    DM_minus = np.zeros(N, dtype=np.float64)
    
    for i in range(1, N):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        
        if up_move > down_move and up_move > 0:
            DM_plus[i] = up_move
        else:
            DM_plus[i] = 0.0
        
        if down_move > up_move and down_move > 0:
            DM_minus[i] = down_move
        else:
            DM_minus[i] = 0.0
    
    # Step 2: 平滑TR, DM+, DM- (EMA-like Smoothing)
    # EMA Smoothing Formula: Smoothed X_i = (Smoothed X_{i-1} * (period - 1) + X_i) / period
    
    smooth_TR = np.zeros(N, dtype=np.float64)
    smooth_DM_plus = np.zeros(N, dtype=np.float64)
    smooth_DM_minus = np.zeros(N, dtype=np.float64)
    
    # 初始平滑值（从索引period-1开始）
    if period > 1:
        smooth_TR[period - 1] = np.sum(TR[1:period]) / (period - 1)
        smooth_DM_plus[period - 1] = np.sum(DM_plus[1:period]) / (period - 1)
        smooth_DM_minus[period - 1] = np.sum(DM_minus[1:period]) / (period - 1)
    
    # 后续平滑
    for i in range(period, N):
        smooth_TR[i] = (smooth_TR[i-1] * (period - 1) + TR[i]) / period
        smooth_DM_plus[i] = (smooth_DM_plus[i-1] * (period - 1) + DM_plus[i]) / period
        smooth_DM_minus[i] = (smooth_DM_minus[i-1] * (period - 1) + DM_minus[i]) / period
    
    # Step 3: 计算DI+和DI-
    DI_plus = np.zeros(N, dtype=np.float64)
    DI_minus = np.zeros(N, dtype=np.float64)
    
    for i in range(period - 1, N):
        if smooth_TR[i] > 0:
            DI_plus[i] = (smooth_DM_plus[i] / smooth_TR[i]) * 100.0
            DI_minus[i] = (smooth_DM_minus[i] / smooth_TR[i]) * 100.0
        else:
            DI_plus[i] = 0.0
            DI_minus[i] = 0.0
    
    # Step 4: 计算DX (Directional Movement Index)
    DX = np.zeros(N, dtype=np.float64)
    
    for i in range(period - 1, N):
        DI_sum = DI_plus[i] + DI_minus[i]
        if DI_sum > 0:
            DX[i] = (np.abs(DI_plus[i] - DI_minus[i]) / DI_sum) * 100.0
        else:
            DX[i] = 0.0
    
    # Step 5: 平滑DX得到ADX
    ADX = np.zeros(N, dtype=np.float64)
    
    # ADX的第一个值从2*period-1开始
    if N >= 2 * period - 1:
        # 初始ADX（使用SMA）
        ADX[2 * period - 2] = np.sum(DX[period - 1:2 * period - 1]) / period
        
        # 后续平滑（EMA-like Smoothing）
        for i in range(2 * period - 1, N):
            ADX[i] = (ADX[i-1] * (period - 1) + DX[i]) / period
    
    return ADX


# --- 4. RSI (相对强弱指标) JIT 函数 ---

@njit(float64[:](float64[:], int64))
def calculate_rsi_jit(close: np.ndarray, period: int) -> np.ndarray:
    """
    Numba JIT编译的RSI (Relative Strength Index) 计算
    
    Args:
        close: 收盘价数组
        period: RSI计算周期（通常14）
        
    Returns:
        RSI值数组（0-100）
    """
    N = len(close)
    if N < period + 1:
        return np.array([50.0])  # 默认中性值
    
    RSI = np.zeros(N, dtype=np.float64)
    
    # 计算价格变化
    deltas = np.zeros(N, dtype=np.float64)
    for i in range(1, N):
        deltas[i] = close[i] - close[i-1]
    
    # 计算初始平均收益和平均损失
    gains = np.zeros(N, dtype=np.float64)
    losses = np.zeros(N, dtype=np.float64)
    
    for i in range(1, N):
        if deltas[i] > 0:
            gains[i] = deltas[i]
        else:
            losses[i] = -deltas[i]
    
    # 计算初始平均收益和平均损失（SMA）
    avg_gain = 0.0
    avg_loss = 0.0
    
    for i in range(1, period + 1):
        avg_gain += gains[i]
        avg_loss += losses[i]
    
    avg_gain /= period
    avg_loss /= period
    
    # 计算初始RSI
    if avg_loss == 0.0:
        RSI[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        RSI[period] = 100.0 - (100.0 / (1.0 + rs))
    
    # 使用EMA平滑方法计算后续RSI
    for i in range(period + 1, N):
        # EMA平滑
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
        if avg_loss == 0.0:
            RSI[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            RSI[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return RSI


# --- 3. 宏观上下文管理器 ---

class MacroContext:
    """
    用于L2 FSM线程存储和更新宏观指标的内存结构。
    
    职责：
    1. 维护K线历史数据
    2. 实时计算ATR、BBANDS等宏观指标
    3. 提供指标访问接口
    """
    
    def __init__(self, history_size: int, config_params: Dict[str, Union[int, float]]):
        """
        初始化宏观上下文
        
        Args:
            history_size: K线历史数据大小（默认2880，支持2天M1 K线：2天×1440分钟=2880根）
            config_params: 配置参数字典（从ConfigManager获取）
        """
        # 【数据层优化】支持2天M1 K线历史（2880根）
        # 2天 × 1440分钟/天 = 2880根K线
        # 确保所有宏观指标（ATR、BBANDS、ADX等）有足够的历史深度
        self.history_size = history_size if history_size >= 2880 else 2880
        self.config = config_params
        
        # 存储K线历史数据（NumPy数组的deque，避免在每次收盘时重建大数组）
        # maxlen=2880确保自动维护2天历史，自动丢弃最旧数据
        self.kline_history: Deque[np.ndarray] = deque(maxlen=self.history_size)
        
            # 实时指标输出
        self.current_atr: float = 0.0
        self.current_bb_upper: float = 0.0
        self.current_bb_lower: float = 0.0
        self.current_bb_mid: float = 0.0
        self.current_adx: float = 0.0
        self.current_rsi: float = 50.0  # RSI默认值50（中性）
        
        # 微观动能刷单指标
        self.current_momentum_delta: float = 0.0  # 动量指标 ΔP
        self.current_decay_long: float = 0.5  # 多头动能衰竭指标（判断空头动能衰竭）
        self.current_decay_short: float = 0.5  # 空头动能衰竭指标（判断多头动能衰竭）
        self.prev_kline: Optional[np.ndarray] = None  # 上一根K线（用于计算动量）

    def load_historical_klines(self, klines: list):
        """
        从Redis或MT5加载历史K线数据
        
        Args:
            klines: K线数据列表，每个元素为字典格式：
                {'time': int, 'open': float, 'high': float, 'low': float, 'close': float, 'volume': int}
        """
        if not klines:
            return
        
        # 清空现有历史
        self.kline_history.clear()
        
        # 转换为NumPy结构化数组并添加到历史
        for kline_dict in klines:
            kline_array = np.array([(
                int(kline_dict['time']),
                float(kline_dict['open']),
                float(kline_dict['high']),
                float(kline_dict['low']),
                float(kline_dict['close']),
                int(kline_dict.get('volume', 0))
            )], dtype=KLINE_DTYPE)
            self.kline_history.append(kline_array[0])
        
        # 如果有足够的历史数据，计算指标
        if len(self.kline_history) >= 2:
            full_history = np.array(list(self.kline_history), dtype=KLINE_DTYPE)
            atr_period = int(self.config.get('HISTORY_CANDLES_N', 20))
            
            # 计算所有指标
            atr_results = calculate_atr_jit(
                full_history['high'],
                full_history['low'],
                full_history['close'],
                atr_period
            )
            self.current_atr = atr_results[-1] if len(atr_results) > 0 else 0.0
            
            bb_mult = float(self.config.get('BBANDS_SD_MULTIPLIER', 2.0))
            ma, upper, lower = calculate_bbands_jit(full_history['close'], atr_period, bb_mult)
            if len(ma) > 0:
                self.current_bb_mid = ma[-1]
                self.current_bb_upper = upper[-1]
                self.current_bb_lower = lower[-1]
            
            adx_results = calculate_adx_jit(
                full_history['high'],
                full_history['low'],
                full_history['close'],
                atr_period
            )
            self.current_adx = adx_results[-1] if len(adx_results) > 0 and adx_results[-1] > 0 else 0.0
            
            rsi_period = 14
            rsi_results = calculate_rsi_jit(full_history['close'], rsi_period)
            self.current_rsi = rsi_results[-1] if len(rsi_results) > 0 and rsi_results[-1] > 0 else 50.0

    def update_context(self, new_kline: np.ndarray):
        """
        在K线收盘时调用，更新所有宏观指标。
        
        Args:
            new_kline: 新闭合的K线NumPy数组（KLINE_DTYPE格式）
        """
        if not new_kline.size:
            return
        
        # 1. 更新K线历史
        self.kline_history.append(new_kline[0])
        
        # 2. 构造计算所需的NumPy数组
        if len(self.kline_history) < 2:
            return  # 至少需要两根K线才能计算TR
        
        # 构造一个包含所有历史数据的N x 6数组
        full_history = np.array(list(self.kline_history), dtype=KLINE_DTYPE)
        
        # 3. 计算ATR
        atr_period = int(self.config.get('HISTORY_CANDLES_N', 20))  # 使用HISTORY_CANDLES_N作为ATR周期
        
        atr_results = calculate_atr_jit(
            full_history['high'], 
            full_history['low'], 
            full_history['close'], 
            atr_period
        )
        self.current_atr = atr_results[-1] if len(atr_results) > 0 else 0.0
        
        # 4. 计算BBANDS
        bb_mult = float(self.config.get('BBANDS_SD_MULTIPLIER', 2.0))
        
        ma, upper, lower = calculate_bbands_jit(
            full_history['close'], 
            atr_period,  # 周期与ATR相同
            bb_mult
        )
        
        if len(ma) > 0:
            self.current_bb_mid = ma[-1]
            self.current_bb_upper = upper[-1]
            self.current_bb_lower = lower[-1]
        
        # 5. 计算ADX
        adx_results = calculate_adx_jit(
            full_history['high'],
            full_history['low'],
            full_history['close'],
            atr_period
        )
        self.current_adx = adx_results[-1] if len(adx_results) > 0 and adx_results[-1] > 0 else 0.0
        
        # 6. 计算RSI
        rsi_period = 14  # RSI标准周期
        rsi_results = calculate_rsi_jit(full_history['close'], rsi_period)
        self.current_rsi = rsi_results[-1] if len(rsi_results) > 0 and rsi_results[-1] > 0 else 50.0
        
        # 7. 计算微观动能指标（用于刷单模块）
        self._calculate_scalping_momentum(new_kline)
        
        # 保存当前K线作为下一根K线的prev_kline
        self.prev_kline = new_kline[0].copy() if new_kline.size > 0 else None
    
    def get_atr(self) -> float:
        """获取当前ATR值"""
        return self.current_atr
    
    def get_bbands(self) -> Tuple[float, float, float]:
        """
        获取当前布林带值
        
        Returns:
            (上轨, 中轨, 下轨)
        """
        return (self.current_bb_upper, self.current_bb_mid, self.current_bb_lower)
    
    def get_adx(self) -> float:
        """获取当前ADX值"""
        return self.current_adx
    
    def _calculate_scalping_momentum(self, new_kline: np.ndarray):
        """
        计算微观动能指标（用于刷单模块）
        
        1. 动量指标 ΔP = |Close_current - Close_prev| / ATR_short
        2. 动能衰竭指标 Decay_Long = (Close - Low) / (High - Low)  # 判断空头动能衰竭
        3. 动能衰竭指标 Decay_Short = (High - Close) / (High - Low)  # 判断多头动能衰竭
        """
        if not new_kline.size or self.prev_kline is None:
            return
        
        current_close = new_kline[0]['close']
        current_high = new_kline[0]['high']
        current_low = new_kline[0]['low']
        prev_close = self.prev_kline['close']
        
        # 1. 计算动量指标 ΔP
        atr_short = self.current_atr if self.current_atr > 0 else 0.01  # 避免除零
        price_change = abs(current_close - prev_close)
        self.current_momentum_delta = price_change / atr_short if atr_short > 0 else 0.0
        
        # 2. 计算动能衰竭指标
        kline_range = current_high - current_low
        if kline_range > 0:
            # Decay_Long: 收盘价在K线振幅中的位置（判断空头动能衰竭）
            self.current_decay_long = (current_close - current_low) / kline_range
            
            # Decay_Short: 收盘价在K线振幅中的位置（判断多头动能衰竭）
            self.current_decay_short = (current_high - current_close) / kline_range
        else:
            self.current_decay_long = 0.5
            self.current_decay_short = 0.5
    
    def get_rsi(self) -> float:
        """获取当前RSI值"""
        return self.current_rsi
    
    def get_momentum_delta(self) -> float:
        """获取动量指标 ΔP"""
        return self.current_momentum_delta
    
    def get_decay_long(self) -> float:
        """获取多头动能衰竭指标（判断空头动能衰竭，值越高表示空头动能越弱）"""
        return self.current_decay_long
    
    def get_decay_short(self) -> float:
        """获取空头动能衰竭指标（判断多头动能衰竭，值越高表示多头动能越弱）"""
        return self.current_decay_short
    
    def get_kline_count(self) -> int:
        """获取当前K线历史数量"""
        return len(self.kline_history)
    
    def reset(self):
        """重置所有历史数据（用于模式切换时清理状态）"""
        self.kline_history.clear()
        self.current_atr = 0.0
        self.current_bb_upper = 0.0
        self.current_bb_lower = 0.0
        self.current_bb_mid = 0.0
        self.current_adx = 0.0
        self.current_rsi = 50.0

