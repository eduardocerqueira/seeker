#date: 2025-11-13T17:03:06Z
#url: https://api.github.com/gists/8dde3cf7f589cb5afb9a45bd262da41a
#owner: https://api.github.com/users/wangwei334455

"""
震荡模式策略：通道反转（Mean Reversion）
目标：高抛低吸
"""
from typing import Optional, Dict, Any
from loguru import logger

from .base_strategy import BaseStrategy, Signal, MarketMode
from src.trading.core.indicators.micro_indicators import MicroContext
from src.trading.core.indicators.macro_indicators import MacroContext


class RangingStrategy(BaseStrategy):
    """
    震荡模式策略：通道反转
    
    策略逻辑：
    1. 价格触及布林带上轨 + LRS反转信号 → 卖出
    2. 价格触及布林带下轨 + LRS反转信号 → 买入
    3. 价格回到中轴 → 平仓
    4. TICK密度爆发 → 止损（假突破过滤）
    """
    
    def __init__(self, config_manager, symbol: str = "BTCUSDm"):
        super().__init__(config_manager, symbol)
        
        # 加载配置参数
        self._load_config()
        
        # 初始化指标上下文（这些应该由L2 FSM传入，这里简化）
        self.micro_context: Optional[MicroContext] = None
        self.macro_context: Optional[MacroContext] = None
        
        # 策略状态
        self.last_signal = Signal.NONE
    
    def _load_config(self):
        """加载配置参数"""
        self.bbands_sd_mult = float(self.get_config('RANGING', 'BBANDS_SD_MULTIPLIER', 2.0))
        self.lrs_reverse_threshold = float(self.get_config('RANGING', 'LRS_REVERSE_THRESHOLD', 0.00005))
        self.density_breakout_mult = float(self.get_config('RANGING', 'DENSITY_BREAKOUT_MULTIPLIER', 3.0))
        self.atr_sl_mult = float(self.get_config('RANGING', 'ATR_SL_MULTIPLIER', 2.0))
        self.exit_tolerance = float(self.get_config('RANGING', 'EXIT_TARGET_TOLERANCE', 0.0001))
    
    def set_contexts(self, micro_context: MicroContext, macro_context: MacroContext):
        """
        设置指标上下文（由L2 FSM调用）
        
        Args:
            micro_context: 微观指标上下文
            macro_context: 宏观指标上下文
        """
        self.micro_context = micro_context
        self.macro_context = macro_context
    
    def on_tick(self, tick_data: Dict[str, Any]) -> Optional[Signal]:
        """
        TICK数据回调
        
        Args:
            tick_data: TICK数据字典
            
        Returns:
            交易信号
        """
        if not self.micro_context or not self.macro_context:
            return Signal.NONE
        
        # 获取当前价格
        current_price = tick_data.get('last', tick_data.get('bid', 0.0))
        if current_price == 0.0:
            return Signal.NONE
        
        # 获取指标值
        lrs = self.micro_context.get_lrs()
        density = self.micro_context.get_density()
        atr = self.macro_context.get_atr()
        bb_upper, bb_mid, bb_lower = self.macro_context.get_bbands()
        
        # 计算平均TICK密度（简化：使用当前值作为基准）
        avg_density = density  # 实际应该计算历史平均值
        
        # 策略逻辑：通道反转
        signal = self._generate_ranging_signal(
            current_price, lrs, density, avg_density,
            bb_upper, bb_mid, bb_lower, atr
        )
        
        self.last_signal = signal
        return signal
    
    def _generate_ranging_signal(self, price: float, lrs: float, density: int,
                                 avg_density: int, bb_upper: float, bb_mid: float,
                                 bb_lower: float, atr: float) -> Signal:
        """
        生成震荡模式交易信号
        
        Args:
            price: 当前价格
            lrs: LRS斜率
            density: TICK密度
            avg_density: 平均TICK密度
            bb_upper: 布林带上轨
            bb_mid: 布林带中轨
            bb_lower: 布林带下轨
            atr: ATR值
            
        Returns:
            交易信号
        """
        # 1. 检查TICK密度爆发（假突破过滤）
        if density > (avg_density * self.density_breakout_mult):
            # TICK密度爆发，可能是真突破，不执行反转交易
            logger.debug(f"RangingStrategy: TICK密度爆发，跳过反转交易 (density={density}, avg={avg_density})")
            return Signal.HOLD
        
        # 2. 检查LRS反转信号
        lrs_abs = abs(lrs)
        if lrs_abs > self.lrs_reverse_threshold:
            # LRS未反转，继续持仓或等待
            return Signal.HOLD
        
        # 3. 价格触及上轨 + LRS反转 → 卖出
        if price >= bb_upper * (1 - self.exit_tolerance):
            if lrs_abs < self.lrs_reverse_threshold:
                logger.info(f"RangingStrategy: 卖出信号 - price={price}, bb_upper={bb_upper}, lrs={lrs}")
                return Signal.SELL
        
        # 4. 价格触及下轨 + LRS反转 → 买入
        if price <= bb_lower * (1 + self.exit_tolerance):
            if lrs_abs < self.lrs_reverse_threshold:
                logger.info(f"RangingStrategy: 买入信号 - price={price}, bb_lower={bb_lower}, lrs={lrs}")
                return Signal.BUY
        
        # 5. 价格回到中轴 → 平仓
        if abs(price - bb_mid) < (bb_mid * self.exit_tolerance):
            if self.last_signal in (Signal.BUY, Signal.SELL):
                logger.info(f"RangingStrategy: 平仓信号 - price={price}, bb_mid={bb_mid}")
                return Signal.CLOSE
        
        return Signal.HOLD
    
    def on_kline(self, kline_data: Dict[str, Any]) -> None:
        """
        K线收盘回调
        
        Args:
            kline_data: K线数据
        """
        # K线收盘时，宏观指标已经由MacroContext更新
        # 这里可以执行一些基于K线的额外逻辑
        pass
    
    def generate_signal(self) -> Optional[Signal]:
        """
        生成交易信号（综合方法）
        
        Returns:
            交易信号
        """
        return self.last_signal
    
    def on_mode_switch(self, new_mode: MarketMode) -> None:
        """
        模式切换回调（平仓逻辑）
        
        Args:
            new_mode: 新的市场模式
        """
        logger.info(f"RangingStrategy: 模式切换 - 从 RANGING 切换到 {new_mode.name}")
        
        # 执行平仓逻辑
        if self.positions:
            logger.info(f"RangingStrategy: 平仓所有持仓 - {len(self.positions)} 个")
            self.positions.clear()
        
        self.last_signal = Signal.NONE

