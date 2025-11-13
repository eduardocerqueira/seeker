#date: 2025-11-13T17:03:06Z
#url: https://api.github.com/gists/8dde3cf7f589cb5afb9a45bd262da41a
#owner: https://api.github.com/users/wangwei334455

"""
下跌趋势模式策略：趋势跟踪 (Trend Following)

入场条件：ADX确认强趋势 AND LRS负向动能健康 OR 价格短暂反弹
"""
from typing import Optional, Dict, Any
from loguru import logger

from .base_strategy import BaseStrategy, Signal, MarketMode
from src.trading.core.indicators.micro_indicators import MicroContext
from src.trading.core.indicators.macro_indicators import MacroContext


class DowntrendStrategy(BaseStrategy):
    """
    下跌趋势模式策略：趋势跟踪
    
    策略逻辑：
    1. ADX确认强趋势（ADX > 阈值）
    2. LRS负向动能（LRS < 负阈值）→ 卖出
    3. 价格反弹到中轨附近 → 卖出
    4. 移动止损（TSL）
    """
    
    def __init__(self, config_manager, symbol: str = "BTCUSDm"):
        super().__init__(config_manager, symbol)
        
        # 加载配置参数
        self._load_config()
        
        # 初始化指标上下文（由L2 FSM传入）
        self.micro_context: Optional[MicroContext] = None
        self.macro_context: Optional[MacroContext] = None
        
        # 策略状态
        self.last_signal = Signal.NONE
    
    def _load_config(self):
        """加载配置参数"""
        self.adx_min_threshold = float(self.get_config('DOWNTREND', 'ADX_MIN_THRESHOLD', 30.0))
        self.lrs_min_momentum = float(self.get_config('DOWNTREND', 'LRS_MIN_MOMENTUM', 0.00015))  # 注意：是正值，但用于比较负值
        self.density_min_mult = float(self.get_config('DOWNTREND', 'DENSITY_MIN_MULTIPLIER', 1.5))
        self.tsl_atr_mult = float(self.get_config('DOWNTREND', 'TSL_ATR_MULTIPLIER', 3.0))
        self.tsl_ma_period = int(self.get_config('DOWNTREND', 'TSL_MA_PERIOD', 10))
    
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
        
        # 获取当前价格（使用ASK价格卖出）
        current_price = tick_data.get('ask', tick_data.get('last', 0.0))
        if current_price == 0.0:
            return Signal.NONE
        
        # 获取指标值
        lrs = self.micro_context.get_lrs()
        density = self.micro_context.get_density()
        adx = self.macro_context.get_adx()
        bb_mid = self.macro_context.get_bb_mid()
        atr = self.macro_context.get_atr()
        
        # 1. 检查出场条件（移动止损）
        exit_signal = self._check_exit_conditions(current_price, bb_mid, atr)
        if exit_signal != Signal.NONE:
            return exit_signal
        
        # 2. 检查开仓条件（追空）
        
        # a. 宏观趋势确认
        if adx < self.adx_min_threshold:
            # 趋势不够强，不进行交易
            return Signal.NONE
        
        # b. 微观动能和入场判断
        
        # 策略1: 强动能突破（LRS快速下降）
        if lrs <= -self.lrs_min_momentum:  # 注意：LRS必须是负值
            # 价格应该在布林中轨之下
            if current_price < bb_mid:
                logger.info(f"DowntrendStrategy: 卖出信号 - price={current_price:.4f} < mid={bb_mid:.4f}, LRS={lrs:.6f} (强负动能)")
                return Signal.SELL
        
        # 策略2: 短暂反弹卖出（价格回到中轨附近，但LRS仍为负）
        if current_price < bb_mid * 1.005 and lrs < 0:
            # 检查TICK密度（确保不是假突破）
            avg_density = density  # 简化：使用当前值
            if density >= (avg_density * self.density_min_mult):
                logger.info(f"DowntrendStrategy: 反弹卖出信号 - price={current_price:.4f}, LRS={lrs:.6f}")
                return Signal.SELL
        
        return Signal.HOLD
    
    def _check_exit_conditions(self, current_price: float, bb_mid: float, atr: float) -> Signal:
        """
        检查出场条件（移动止损）
        
        Args:
            current_price: 当前价格
            bb_mid: 布林带中轨
            atr: ATR值
            
        Returns:
            出场信号
        """
        if not self.positions:
            return Signal.NONE
        
        # 检查移动止损：价格突破TSL MA或ATR止损
        for pos_id, position in self.positions.items():
            entry_price = position.get('price', 0.0)
            if entry_price == 0.0:
                continue
            
            # TSL方法1: 价格突破中轨（简化）
            if current_price > bb_mid:
                logger.info(f"DowntrendStrategy: 移动止损触发 - price={current_price:.4f} > mid={bb_mid:.4f}")
                return Signal.CLOSE
            
            # TSL方法2: ATR止损（价格突破入场价 + TSL_ATR_MULTIPLIER * ATR）
            tsl_price = entry_price + (atr * self.tsl_atr_mult)
            if current_price > tsl_price:
                logger.info(f"DowntrendStrategy: ATR止损触发 - price={current_price:.4f} > tsl={tsl_price:.4f}")
                return Signal.CLOSE
        
        return Signal.NONE
    
    def on_kline(self, kline_data: Dict[str, Any]) -> None:
        """
        K线收盘回调
        
        检查趋势是否终结，以便FSM切换模式
        
        Args:
            kline_data: K线数据
        """
        if not self.macro_context:
            return
        
        # 检查ADX是否衰减（趋势终结预警）
        adx_max_ranging = float(self.get_config('RANGING', 'ADX_MAX_THRESHOLD', 25.0))
        
        if self.macro_context.get_adx() < adx_max_ranging:
            logger.debug(f"DowntrendStrategy: ADX {self.macro_context.get_adx():.2f} < Threshold {adx_max_ranging}. 趋势减弱")
            # FSM核心的宏观守护逻辑将据此进行模式切换
    
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
        logger.info(f"DowntrendStrategy: 模式切换 - 从 DOWNTREND 切换到 {new_mode.name}")
        
        # 执行平仓逻辑
        if self.positions:
            logger.info(f"DowntrendStrategy: 平仓所有持仓 - {len(self.positions)} 个")
            self.positions.clear()
        
        self.last_signal = Signal.NONE

