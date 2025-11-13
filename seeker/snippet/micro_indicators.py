#date: 2025-11-13T17:03:06Z
#url: https://api.github.com/gists/8dde3cf7f589cb5afb9a45bd262da41a
#owner: https://api.github.com/users/wangwei334455

"""
微观指标计算模块：包含TICK级别的高速JIT编译函数。

- LRS (线性回归斜率)
- TICK Density (波动频率)
"""
import numpy as np
from numba import njit, float64, int64
from collections import deque
from typing import Deque


# --- 1. LRS (线性回归斜率) JIT 函数 ---

@njit(float64(float64[:]))
def calculate_lrs_slope_jit(prices: np.ndarray) -> float:
    """
    Numba JIT编译的线性回归斜率计算。
    
    x是时间索引 (0, 1, 2, ... N-1)，y是价格。
    
    Args:
        prices: 价格数组（NumPy数组）
        
    Returns:
        线性回归斜率值
    """
    N = len(prices)
    if N < 2:
        return 0.0

    # 优化: 避免使用sum()等高层函数，使用循环提高JIT优化效果
    x_sum = 0.0
    y_sum = 0.0
    sum_xy = 0.0
    sum_x_squared = 0.0
    
    for i in range(N):
        x_val = float64(i)
        y_val = prices[i]
        
        x_sum += x_val
        y_sum += y_val
        sum_xy += x_val * y_val
        sum_x_squared += x_val * x_val

    # 分子和分母
    numerator = N * sum_xy - x_sum * y_sum
    denominator = N * sum_x_squared - x_sum * x_sum
    
    if denominator == 0.0:
        return 0.0
    
    return numerator / denominator


# --- 2. MicroContext 管理器 ---

class MicroContext:
    """
    用于L2 FSM线程存储和更新微观指标的内存结构。
    
    职责：
    1. 维护TICK价格和时间戳缓冲区（确保数据一致性）
    2. 实时计算LRS（线性回归斜率）
    3. 实时计算TICK密度（波动频率）
    
    性能优化：
    - TICK缓存：3000个TICK（环形缓冲区，自动丢弃最旧数据）
    - LRS计算：使用Numba JIT编译，确保<0.1ms性能
    - 数据一致性：时间和价格使用联合数据结构，确保索引关联
    """
    
    def __init__(self, lrs_period: int, density_period_ms: int, tick_cache_size: int = 3000):
        """
        初始化微观上下文
        
        Args:
            lrs_period: LRS计算周期（TICK数量，通常20-50）
            density_period_ms: TICK密度计算的时间窗口（毫秒，通常500ms）
            tick_cache_size: TICK缓存大小（默认3000，覆盖10分钟-1分钟行情）
        """
        self.lrs_period = lrs_period
        self.density_period_ms = density_period_ms
        self.tick_cache_size = tick_cache_size
        
        # 【关键优化】使用联合数据结构确保时间和价格的一致性
        # 使用deque存储(time_msc, price)元组，确保索引关联
        self.tick_buffer: Deque[tuple] = deque(maxlen=tick_cache_size)
        
        # 实时指标输出
        self.current_lrs: float = 0.0
        self.current_density: int = 0
        self.current_momentum: float = 0.0  # 价格变化速度（动量）
        self.avg_tick_density: float = 100.0  # 平均TICK密度（TICK/秒）
        
        # 预分配NumPy数组，用于传入JIT函数，避免在热路径中重复分配内存
        self._price_array = np.zeros(lrs_period, dtype=np.float64)
        
        # 动量计算：存储最近N个价格变化（用于计算短期动量）
        self.momentum_window = 5  # 动量窗口：最近5个价格变化
        self.price_change_buffer: Deque[float] = deque(maxlen=self.momentum_window)  # 存储价格变化率
        self.last_price: float = 0.0

    def update_context(self, time_msc: int, price: float):
        """
        在每个TICK到来时调用，增量更新所有微观指标。
        
        【性能要求】此函数必须在单线程事件循环中执行，确保原子性和确定性。
        所有计算必须在单次循环中完成，禁止阻塞操作。
        
        Args:
            time_msc: TICK时间戳（毫秒）
            price: TICK价格
        """
        # 1. 更新缓冲区 (O(1)) - 使用联合数据结构确保一致性
        self.tick_buffer.append((time_msc, price))
        
        # 2. 计算LRS斜率（Numba JIT优化，目标<0.1ms）
        self.current_lrs = self._calculate_current_lrs()
        
        # 3. 计算TICK密度（增量计算，避免遍历）
        self.current_density = self._calculate_current_density(time_msc)
        
        # 4. 更新平均TICK密度（用于突破预警）
        self._update_avg_density()
        
        # 5. 计算动量（价格变化速度）
        if self.last_price > 0:
            price_change = (price - self.last_price) / self.last_price if self.last_price > 0 else 0.0
            self.price_change_buffer.append(price_change)
            self.current_momentum = self._calculate_momentum()
        self.last_price = price

    def _calculate_current_lrs(self) -> float:
        """
        将价格数据传入JIT函数计算LRS。
        
        【性能优化】使用预分配数组和Numba JIT编译，确保<0.1ms性能。
        
        Returns:
            当前LRS斜率值（归一化后的斜率，除以平均价格）
        """
        if len(self.tick_buffer) < 2:
            # 至少需要2个点才能计算斜率
            return 0.0
        
        # 获取最近N个TICK的价格（从联合缓冲区提取）
        actual_period = min(len(self.tick_buffer), self.lrs_period)
        
        # 【性能优化】直接填充预分配数组，避免列表推导式
        if actual_period < self.lrs_period:
            # 缓冲区未满，创建临时数组
            prices = np.zeros(actual_period, dtype=np.float64)
            for i in range(actual_period):
                prices[i] = self.tick_buffer[-(actual_period-i)][1]
            return calculate_lrs_slope_jit(prices)
        
        # 缓冲区已满，使用预分配数组（零拷贝/低开销）
        # 从后往前提取价格（最新数据在末尾）
        for i in range(actual_period):
            self._price_array[i] = self.tick_buffer[-(actual_period-i)][1]
        
        slope = calculate_lrs_slope_jit(self._price_array)
        
        # 归一化：除以平均价格，使LRS值在不同价格水平下可比较
        avg_price = np.mean(self._price_array[:actual_period])
        normalized_slope = slope / avg_price if avg_price > 0 else 0.0
        
        return normalized_slope

    def _calculate_current_density(self, current_time_ms: int) -> int:
        """
        计算指定毫秒窗口内接收到的TICK数量（TICK密度）。
        
        【性能优化】从联合缓冲区提取时间戳，使用增量计算避免全量遍历。
        
        Args:
            current_time_ms: 当前时间戳（毫秒）
            
        Returns:
            时间窗口内的TICK数量
        """
        time_window_start = current_time_ms - self.density_period_ms
        
        count = 0
        # 从联合缓冲区提取时间戳，计算在窗口内的数量
        # 由于deque是FIFO，从后往前遍历（最新数据在末尾）
        for time_msc, _ in reversed(self.tick_buffer):
            if time_msc >= time_window_start:
                count += 1
            else:
                # 由于数据按时间排序，一旦遇到超出窗口的数据，后续数据也必然超出
                break
        
        return count
    
    def _update_avg_density(self):
        """
        更新平均TICK密度（用于突破预警判断）
        
        使用指数移动平均（EMA）平滑处理，避免突发波动影响
        """
        if self.current_density > 0:
            # EMA平滑：alpha=0.1，给予新值10%权重
            alpha = 0.1
            self.avg_tick_density = alpha * self.current_density + (1 - alpha) * self.avg_tick_density
    
    def _calculate_momentum(self) -> float:
        """
        计算价格变化速度（动量）
        
        动量 = 最近N个价格变化的平均值（绝对值）
        用于检测价格是否在短时间内快速移动
        
        Returns:
            动量值（绝对值，越大表示价格变化越快）
        """
        if len(self.price_change_buffer) < 2:
            return 0.0
        
        # 计算最近N个价格变化的平均绝对值
        momentum = sum(abs(change) for change in self.price_change_buffer) / len(self.price_change_buffer)
        return momentum
    
    def get_lrs(self) -> float:
        """获取当前LRS值"""
        return self.current_lrs
    
    def get_density(self) -> int:
        """获取当前TICK密度"""
        return self.current_density
    
    def get_momentum(self) -> float:
        """获取当前动量值（价格变化速度）"""
        return self.current_momentum
    
    def get_avg_density(self) -> float:
        """获取平均TICK密度（TICK/秒）"""
        return self.avg_tick_density
    
    def reset(self):
        """重置所有缓冲区（用于模式切换时清理状态）"""
        self.tick_buffer.clear()
        self.price_change_buffer.clear()
        self.current_lrs = 0.0
        self.current_density = 0
        self.current_momentum = 0.0
        self.avg_tick_density = 100.0
        self.last_price = 0.0

