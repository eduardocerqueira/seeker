#date: 2025-11-13T17:02:59Z
#url: https://api.github.com/gists/c1c2c32c02ad6b73f59da7fc63add2f0
#owner: https://api.github.com/users/wangwei334455

"""
风控管理器：执行下单前的硬性风控检查

【职责】
1. 仓位限制检查
2. 最大亏损检查
3. 交易频率限制
4. 单笔订单风险检查
"""
import time
import logging
from typing import Dict, Any, Optional
from collections import deque
from threading import Lock

logger = logging.getLogger(__name__)


class RiskController:
    """
    实时风控管理器
    
    【风控规则】
    1. 最大仓位限制（单方向/总仓位）
    2. 最大日内亏损限制
    3. 交易频率限制（防止过度交易）
    4. 单笔订单风险限制（基于ATR）
    """
    
    def __init__(self, symbol: str, config: Optional[Dict[str, Any]] = None):
        """
        初始化风控管理器
        
        Args:
            symbol: 交易品种
            config: 风控配置字典
        """
        self.symbol = symbol
        self.config = config or {}
        
        # 风控参数（可从配置读取）
        self.max_position_size = float(self.config.get('max_position_size', 1.0))  # 最大仓位（手）
        self.max_daily_loss_usd = float(self.config.get('max_daily_loss_usd', 500.0))  # 最大日内亏损（USD）
        self.max_trades_per_minute = int(self.config.get('max_trades_per_minute', 10))  # 每分钟最大交易次数
        self.max_order_risk_atr = float(self.config.get('max_order_risk_atr', 2.0))  # 单笔订单最大风险（ATR倍数）
        
        # 实时状态跟踪
        self.current_position_long = 0.0  # 当前多头仓位
        self.current_position_short = 0.0  # 当前空头仓位
        self.daily_pnl = 0.0  # 日内盈亏
        self.daily_pnl_lock = Lock()
        
        # 交易频率跟踪（滑动窗口）
        self.trade_timestamps = deque(maxlen=100)  # 最近100笔交易的时间戳
        self.frequency_lock = Lock()
        
        # 交易日开始时间（用于重置日内统计）
        self.trading_day_start = time.time()
        
        # 监控服务引用（可选，用于报告风险突破）
        self.monitor_service = None
        
        logger.info(f"风控管理器已初始化: {symbol} | 最大仓位: {self.max_position_size} | 最大亏损: {self.max_daily_loss_usd} USD")
    
    def set_monitor_service(self, monitor_service):
        """设置监控服务引用（用于报告风险突破）"""
        self.monitor_service = monitor_service
    
    def check_pre_trade_limits(self, side: str, quantity: float, current_price: float, atr: Optional[float] = None) -> tuple[bool, str]:
        """
        执行下单前的风控检查
        
        Args:
            side: 交易方向 ('BUY' / 'SELL')
            quantity: 订单数量（手）
            current_price: 当前价格
            atr: 当前ATR值（可选，用于风险计算）
            
        Returns:
            (bool, str): (是否通过, 拒绝原因)
        """
        # 1. 检查最大仓位限制
        if side == 'BUY':
            new_long_position = self.current_position_long + quantity
            if new_long_position > self.max_position_size:
                reason = f"多头仓位超限: {new_long_position:.2f} > {self.max_position_size:.2f}"
                if self.monitor_service:
                    self.monitor_service.report_risk_breach('MAX_POS', new_long_position, self.max_position_size)
                return False, reason
        elif side == 'SELL':
            new_short_position = self.current_position_short + quantity
            if new_short_position > self.max_position_size:
                reason = f"空头仓位超限: {new_short_position:.2f} > {self.max_position_size:.2f}"
                if self.monitor_service:
                    self.monitor_service.report_risk_breach('MAX_POS', new_short_position, self.max_position_size)
                return False, reason
        
        # 2. 检查总仓位限制
        total_position = abs(self.current_position_long) + abs(self.current_position_short)
        if total_position + quantity > self.max_position_size * 2:  # 允许双向持仓
            reason = f"总仓位超限: {total_position + quantity:.2f} > {self.max_position_size * 2:.2f}"
            if self.monitor_service:
                self.monitor_service.report_risk_breach('MAX_POS', total_position + quantity, self.max_position_size * 2)
            return False, reason
        
        # 3. 检查最大日内亏损
        with self.daily_pnl_lock:
            if self.daily_pnl <= -self.max_daily_loss_usd:
                reason = f"日内亏损已达上限: {self.daily_pnl:.2f} USD <= -{self.max_daily_loss_usd} USD"
                if self.monitor_service:
                    self.monitor_service.report_risk_breach('MAX_DD', abs(self.daily_pnl), self.max_daily_loss_usd)
                return False, reason
        
        # 4. 检查交易频率限制
        with self.frequency_lock:
            current_time = time.time()
            # 清理1分钟前的记录
            while self.trade_timestamps and current_time - self.trade_timestamps[0] > 60:
                self.trade_timestamps.popleft()
            
            if len(self.trade_timestamps) >= self.max_trades_per_minute:
                reason = f"交易频率超限: 过去1分钟内已有 {len(self.trade_timestamps)} 笔交易"
                if self.monitor_service:
                    self.monitor_service.report_risk_breach('FREQUENCY', len(self.trade_timestamps), self.max_trades_per_minute)
                return False, reason
        
        # 5. 检查单笔订单风险（基于ATR）
        if atr and atr > 0:
            order_risk_usd = quantity * atr * self.max_order_risk_atr * current_price
            # 假设最大单笔风险为最大日内亏损的10%
            max_single_order_risk = self.max_daily_loss_usd * 0.1
            if order_risk_usd > max_single_order_risk:
                return False, f"单笔订单风险过大: {order_risk_usd:.2f} USD > {max_single_order_risk:.2f} USD"
        
        # 所有检查通过
        return True, "OK"
    
    def update_position(self, side: str, quantity: float):
        """
        更新仓位（订单成交后调用）
        
        Args:
            side: 交易方向 ('BUY' / 'SELL')
            quantity: 成交数量（手）
        """
        if side == 'BUY':
            self.current_position_long += quantity
        elif side == 'SELL':
            self.current_position_short += quantity
        elif side == 'FLAT':
            # 平仓：根据当前仓位方向平仓
            if self.current_position_long > 0:
                self.current_position_long = max(0, self.current_position_long - quantity)
            elif self.current_position_short > 0:
                self.current_position_short = max(0, self.current_position_short - quantity)
        
        logger.debug(f"仓位更新: LONG={self.current_position_long:.2f}, SHORT={self.current_position_short:.2f}")
    
    def update_daily_pnl(self, pnl_delta: float):
        """
        更新日内盈亏
        
        Args:
            pnl_delta: 盈亏变化量（USD）
        """
        with self.daily_pnl_lock:
            self.daily_pnl += pnl_delta
            logger.debug(f"日内盈亏更新: {self.daily_pnl:.2f} USD")
    
    def record_trade(self):
        """记录交易时间戳（用于频率限制）"""
        with self.frequency_lock:
            self.trade_timestamps.append(time.time())
    
    def reset_daily_stats(self):
        """重置日内统计（新交易日开始时调用）"""
        with self.daily_pnl_lock:
            self.daily_pnl = 0.0
        self.trading_day_start = time.time()
        logger.info("日内统计已重置")
    
    def get_risk_status(self) -> Dict[str, Any]:
        """
        获取当前风控状态
        
        Returns:
            风控状态字典
        """
        with self.daily_pnl_lock:
            daily_pnl = self.daily_pnl
        
        with self.frequency_lock:
            recent_trades = len(self.trade_timestamps)
        
        return {
            'position_long': self.current_position_long,
            'position_short': self.current_position_short,
            'total_position': abs(self.current_position_long) + abs(self.current_position_short),
            'daily_pnl': daily_pnl,
            'recent_trades_count': recent_trades,
            'max_position_size': self.max_position_size,
            'max_daily_loss': self.max_daily_loss_usd,
            'risk_utilization': abs(daily_pnl) / self.max_daily_loss_usd if self.max_daily_loss_usd > 0 else 0.0
        }

