#date: 2025-11-13T17:02:59Z
#url: https://api.github.com/gists/c1c2c32c02ad6b73f59da7fc63add2f0
#owner: https://api.github.com/users/wangwei334455

"""
交易信号生成器 (Signal Generator)
负责结合特征数据、账户状态和策略决策，生成最终交易信号

【核心职责】
1. 消费特征数据（来自FeatureEngine）
2. 结合账户状态（来自AccountStateManager）
3. 执行策略逻辑
4. 生成交易信号（BUY, SELL, CLOSE）
5. 风控检查

【设计原则】
- 单一职责：只负责信号生成，不负责执行
- 可组合性：可以组合多个策略逻辑
- 可测试性：纯函数设计，易于单元测试
"""
from typing import Dict, Optional, Any, List
from dataclasses import dataclass
from loguru import logger

from src.trading.services.account_state_manager import AccountStateManager, PositionData


@dataclass
class TradingSignal:
    """交易信号数据结构"""
    action: str                    # 'BUY', 'SELL', 'CLOSE'
    symbol: str                    # 交易品种
    volume: float                  # 交易量（手）
    price: float                   # 委托价格（0=市价单）
    stop_loss: Optional[float] = None      # 止损价格
    take_profit: Optional[float] = None    # 止盈价格
    magic: int = 202409            # 魔术号
    comment: str = ''              # 备注
    reason: str = ''               # 信号原因
    confidence: float = 0.0        # 信号置信度 (0-1)
    timestamp: int = 0              # 时间戳（毫秒）
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'action': self.action,
            'symbol': self.symbol,
            'volume': self.volume,
            'price': self.price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'magic': self.magic,
            'comment': self.comment,
            'reason': self.reason,
            'confidence': self.confidence,
            'timestamp': self.timestamp,
        }


class SignalGenerator:
    """
    交易信号生成器
    
    【信号生成逻辑】
    1. 接收特征数据（市场状态、技术指标）
    2. 检查账户状态（是否有持仓、可用保证金等）
    3. 执行策略逻辑（基于特征和账户状态）
    4. 生成交易信号
    5. 风控检查（仓位限制、风险控制）
    """
    
    def __init__(
        self,
        account_manager: AccountStateManager,
        default_magic: int = 202409,
        default_volume: float = 0.01,
        max_position_size: float = 1.0,
        sl_percent: float = 0.005,  # 止损阈值：0.5%
        tp_percent: float = 0.01,   # 止盈阈值：1.0%
        add_position_pnl_threshold: float = 0.003,  # 加仓阈值：0.3%盈利
    ):
        """
        初始化信号生成器
        
        Args:
            account_manager: 账户状态管理器
            default_magic: 默认魔术号
            default_volume: 默认交易量（手）
            max_position_size: 最大持仓量（手）
            sl_percent: 止损百分比阈值（例如 0.005 = 0.5%）
            tp_percent: 止盈百分比阈值（例如 0.01 = 1.0%）
            add_position_pnl_threshold: 加仓所需的盈利阈值
        """
        self.account_manager = account_manager
        self.default_magic = default_magic
        self.default_volume = default_volume
        self.max_position_size = max_position_size
        self.sl_percent = sl_percent
        self.tp_percent = tp_percent
        self.add_position_pnl_threshold = add_position_pnl_threshold
        
        # 统计信息
        self.stats = {
            'signal_count': 0,
            'buy_signals': 0,
            'sell_signals': 0,
            'close_signals': 0,
            'rejected_signals': 0,
            'stop_loss_triggers': 0,
            'take_profit_triggers': 0,
            'add_position_signals': 0,
        }
        
        logger.info(
            f"交易信号生成器已初始化: "
            f"default_magic={default_magic}, "
            f"止损/止盈: {sl_percent*100:.1f}% / {tp_percent*100:.1f}%, "
            f"default_volume={default_volume}, "
            f"max_position={max_position_size}"
        )
    
    def generate_signal(
        self,
        features: Dict[str, Any],
        strategy_decision: Optional[Dict[str, Any]] = None,
    ) -> Optional[TradingSignal]:
        """
        生成交易信号
        
        【信号生成流程】
        1. 解析特征数据（市场状态、技术指标）
        2. 检查账户状态（持仓、保证金）
        3. 执行策略逻辑（基于特征和账户状态）
        4. 生成信号
        5. 风控检查
        
        Args:
            features: 特征数据（来自FeatureEngine）
            strategy_decision: 策略决策（来自StrategyFSM，可选）
            
        Returns:
            TradingSignal 或 None（如果未生成信号或风控拒绝）
        """
        try:
            symbol = features.get('symbol', 'BTCUSDm')
            market_regime = features.get('market_regime', 'SIDEWAYS')
            entry_signal = features.get('entry_signal')
            close_price = features.get('close_price', 0.0)
            rsi = features.get('rsi_14', 50.0)
            atr = features.get('atr_14', 0.0)
            
            # 1. 检查账户状态
            snapshot = self.account_manager.get_account_snapshot()
            if not snapshot:
                logger.warning("账户快照不可用，无法生成信号")
                return None
            
            # 检查可用保证金
            if snapshot.margin_free < close_price * self.default_volume * 0.1:  # 简单估算
                logger.warning(f"可用保证金不足: {snapshot.margin_free:.2f}")
                return None
            
            # 2. 检查持仓状态
            has_position = self.account_manager.is_position_open(symbol, self.default_magic)
            current_position = self.account_manager.get_position(symbol, self.default_magic)
            
            # 3. 执行策略逻辑
            signal = self._execute_strategy_logic(
                features=features,
                has_position=has_position,
                current_position=current_position,
                strategy_decision=strategy_decision,
            )
            
            if not signal:
                return None
            
            # 4. 风控检查
            if not self._risk_check(signal, current_position):
                self.stats['rejected_signals'] += 1
                logger.warning(f"信号被风控拒绝: {signal.action} {signal.symbol}")
                return None
            
            # 5. 更新统计
            self.stats['signal_count'] += 1
            if signal.action == 'BUY':
                self.stats['buy_signals'] += 1
            elif signal.action == 'SELL':
                self.stats['sell_signals'] += 1
            elif signal.action == 'CLOSE':
                self.stats['close_signals'] += 1
            
            logger.info(
                f"生成交易信号: {signal.action} {signal.symbol} "
                f"{signal.volume}手 @ {signal.price if signal.price > 0 else '市价'} | "
                f"原因: {signal.reason} | 置信度: {signal.confidence:.2f}"
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"生成交易信号失败: {e}")
            return None
    
    def _calculate_pnl_percent(self, current_price: float, position: PositionData) -> float:
        """
        计算浮动盈亏百分比（基于开仓价格）
        
        Args:
            current_price: 当前市场价格
            position: 持仓数据
            
        Returns:
            浮动盈亏百分比（正数=盈利，负数=亏损）
        """
        if position.open_price == 0:
            return 0.0
        
        # 判断持仓方向
        is_long = position.type == 0 if hasattr(position, 'type') else position.side == 'BUY'
        
        # 计算价格变动
        price_change = current_price - position.open_price
        
        if is_long:
            # 多头：价格上涨盈利
            return price_change / position.open_price
        else:
            # 空头：价格下跌盈利
            return -price_change / position.open_price
    
    def _execute_strategy_logic(
        self,
        features: Dict[str, Any],
        has_position: bool,
        current_position: Optional[PositionData],
        strategy_decision: Optional[Dict[str, Any]],
    ) -> Optional[TradingSignal]:
        """
        执行策略逻辑（状态机）
        
        【状态管理】
        - 状态A: 持仓管理逻辑（止盈/止损/加仓）
        - 状态B: 空仓建仓逻辑
        
        Args:
            features: 特征数据
            has_position: 是否有持仓
            current_position: 当前持仓（如果有）
            strategy_decision: 策略决策（可选）
            
        Returns:
            TradingSignal 或 None
        """
        # 状态A: 持仓管理逻辑
        if has_position and current_position:
            return self._manage_position_logic(features, current_position)
        
        # 状态B: 空仓建仓逻辑
        else:
            return self._entry_signal_logic(features)
    
    def _manage_position_logic(
        self,
        features: Dict[str, Any],
        position: PositionData,
    ) -> Optional[TradingSignal]:
        """
        持仓管理逻辑（状态A）
        
        【核心职责】
        1. 计算浮动盈亏
        2. 检查止损（亏损状态）
        3. 检查止盈（盈利状态）
        4. 检查加仓条件（盈利且趋势持续）
        
        Args:
            features: 特征数据
            position: 当前持仓
            
        Returns:
            TradingSignal 或 None
        """
        current_price = features.get('close_price', 0.0)
        market_regime = features.get('market_regime', 'SIDEWAYS')
        atr = features.get('atr_14', 0.0)
        symbol = features.get('symbol', position.symbol)
        
        # 计算浮动盈亏百分比
        pnl_percent = self._calculate_pnl_percent(current_price, position)
        
        # 判断持仓方向
        is_long = position.type == 0 if hasattr(position, 'type') else position.side == 'BUY'
        is_short = position.type == 1 if hasattr(position, 'type') else position.side == 'SELL'
        
        # 平仓方向（与持仓方向相反）
        close_action = 'SELL' if is_long else 'BUY'
        
        logger.debug(
            f"持仓管理: {symbol} | "
            f"方向={'多头' if is_long else '空头'} | "
            f"持仓量={position.volume:.2f}手 | "
            f"开仓价={position.open_price:.2f} | "
            f"当前价={current_price:.2f} | "
            f"盈亏={pnl_percent*100:.2f}%"
        )
        
        # 1. 亏损管理：检查止损（SL）
        if pnl_percent <= -self.sl_percent:
            self.stats['stop_loss_triggers'] += 1
            logger.critical(
                f"🚨 止损触发! {symbol} | "
                f"盈亏={pnl_percent*100:.2f}% <= -{self.sl_percent*100:.1f}%"
            )
            
            return TradingSignal(
                action='CLOSE',
                symbol=symbol,
                volume=position.volume,
                price=0,  # 市价平仓
                magic=position.magic,
                reason=f"止损触发: 盈亏={pnl_percent*100:.2f}%",
                confidence=0.9,
                timestamp=features.get('timestamp', 0),
            )
        
        # 2. 盈利管理：检查止盈（TP）
        if pnl_percent >= self.tp_percent:
            self.stats['take_profit_triggers'] += 1
            logger.success(
                f"🏆 止盈触发! {symbol} | "
                f"盈亏={pnl_percent*100:.2f}% >= {self.tp_percent*100:.1f}%"
            )
            
            return TradingSignal(
                action='CLOSE',
                symbol=symbol,
                volume=position.volume,
                price=0,  # 市价平仓
                magic=position.magic,
                reason=f"止盈触发: 盈亏={pnl_percent*100:.2f}%",
                confidence=0.8,
                timestamp=features.get('timestamp', 0),
            )
        
        # 3. 盈利加仓：盈利且趋势持续
        if (pnl_percent >= self.add_position_pnl_threshold and 
            position.volume < self.max_position_size):
            
            # 检查趋势是否与持仓方向一致
            trend_match = False
            if is_long and market_regime == 'TREND_UP':
                trend_match = True
            elif is_short and market_regime == 'TREND_DOWN':
                trend_match = True
            
            if trend_match:
                # 计算加仓量（不超过最大持仓限制）
                add_volume = min(
                    self.default_volume,
                    self.max_position_size - position.volume
                )
                
                if add_volume > 0:
                    self.stats['add_position_signals'] += 1
                    logger.info(
                        f"📈 盈利加仓: {symbol} | "
                        f"当前持仓={position.volume:.2f}手 | "
                        f"加仓={add_volume:.2f}手 | "
                        f"盈亏={pnl_percent*100:.2f}%"
                    )
                    
                    return TradingSignal(
                        action='BUY' if is_long else 'SELL',
                        symbol=symbol,
                        volume=add_volume,
                        price=0,  # 市价单
                        stop_loss=current_price - 2 * atr if is_long else current_price + 2 * atr,
                        take_profit=current_price + 3 * atr if is_long else current_price - 3 * atr,
                        magic=position.magic,  # 使用相同魔术号，合并持仓
                        reason=f"盈利加仓: 盈亏={pnl_percent*100:.2f}%, 趋势={market_regime}",
                        confidence=0.7,
                        timestamp=features.get('timestamp', 0),
                    )
        
        # 4. 市场状态反转：如果市场状态与持仓方向相反，平仓
        if (is_long and market_regime == 'TREND_DOWN') or \
           (is_short and market_regime == 'TREND_UP'):
            logger.warning(
                f"⚠️ 市场状态反转: {symbol} | "
                f"持仓方向={'多头' if is_long else '空头'} | "
                f"市场状态={market_regime}"
            )
            
            return TradingSignal(
                action='CLOSE',
                symbol=symbol,
                volume=position.volume,
                price=0,
                magic=position.magic,
                reason=f"市场状态反转: 持仓={'多头' if is_long else '空头'}, 市场={market_regime}",
                confidence=0.7,
                timestamp=features.get('timestamp', 0),
            )
        
        # 无操作
        return None
    
    def _entry_signal_logic(
        self,
        features: Dict[str, Any],
    ) -> Optional[TradingSignal]:
        """
        空仓建仓逻辑（状态B）
        
        【核心职责】
        1. 检查市场状态
        2. 检查入场信号
        3. 生成建仓信号
        
        Args:
            features: 特征数据
            
        Returns:
            TradingSignal 或 None
        """
        symbol = features.get('symbol', 'BTCUSDm')
        market_regime = features.get('market_regime', 'SIDEWAYS')
        entry_signal = features.get('entry_signal')
        close_price = features.get('close_price', 0.0)
        rsi = features.get('rsi_14', 50.0)
        atr = features.get('atr_14', 0.0)
        bb_upper = features.get('bb_upper', close_price)
        bb_lower = features.get('bb_lower', close_price)
        
        # 策略1: 极端超卖信号 -> 买入
        if entry_signal == 'EXTREME_OVERSOLD' and market_regime in ['TREND_UP', 'SIDEWAYS']:
            return TradingSignal(
                action='BUY',
                symbol=symbol,
                volume=self.default_volume,
                price=0,  # 市价单
                stop_loss=bb_lower - atr,  # 止损：布林带下轨 - ATR
                take_profit=bb_upper + atr,  # 止盈：布林带上轨 + ATR
                magic=self.default_magic,
                comment='Extreme Oversold Entry',
                reason=f"极端超卖信号: RSI={rsi:.2f}, Regime={market_regime}",
                confidence=0.75,
                timestamp=features.get('timestamp', 0),
            )
        
        # 策略2: 极端超买信号 -> 卖出
        if entry_signal == 'EXTREME_OVERBOUGHT' and market_regime in ['TREND_DOWN', 'SIDEWAYS']:
            return TradingSignal(
                action='SELL',
                symbol=symbol,
                volume=self.default_volume,
                price=0,  # 市价单
                stop_loss=bb_upper + atr,  # 止损：布林带上轨 + ATR
                take_profit=bb_lower - atr,  # 止盈：布林带下轨 - ATR
                magic=self.default_magic,
                comment='Extreme Overbought Entry',
                reason=f"极端超买信号: RSI={rsi:.2f}, Regime={market_regime}",
                confidence=0.75,
                timestamp=features.get('timestamp', 0),
            )
        
        # 策略3: 趋势跟随
        if market_regime == 'TREND_UP' and rsi > 50 and rsi < 70:
            return TradingSignal(
                action='BUY',
                symbol=symbol,
                volume=self.default_volume,
                price=0,
                stop_loss=close_price - 2 * atr,
                take_profit=close_price + 3 * atr,
                magic=self.default_magic,
                comment='Trend Following',
                reason=f"上升趋势跟随: Regime={market_regime}, RSI={rsi:.2f}",
                confidence=0.6,
                timestamp=features.get('timestamp', 0),
            )
        
        if market_regime == 'TREND_DOWN' and rsi < 50 and rsi > 30:
            return TradingSignal(
                action='SELL',
                symbol=symbol,
                volume=self.default_volume,
                price=0,
                stop_loss=close_price + 2 * atr,
                take_profit=close_price - 3 * atr,
                magic=self.default_magic,
                comment='Trend Following',
                reason=f"下降趋势跟随: Regime={market_regime}, RSI={rsi:.2f}",
                confidence=0.6,
                timestamp=features.get('timestamp', 0),
            )
        
        # 如果没有匹配的策略逻辑，返回None
        return None
    
    def _risk_check(
        self,
        signal: TradingSignal,
        current_position: Optional[PositionData],
    ) -> bool:
        """
        风控检查
        
        【风控规则】
        1. 检查仓位限制
        2. 检查可用保证金
        3. 检查信号合理性
        
        Args:
            signal: 交易信号
            current_position: 当前持仓（如果有）
            
        Returns:
            bool: 是否通过风控检查
        """
        try:
            # 1. 检查仓位限制
            if signal.action in ['BUY', 'SELL']:
                total_exposure = self.account_manager.get_total_exposure(signal.symbol)
                if total_exposure + signal.volume > self.max_position_size:
                    logger.warning(
                        f"仓位超限: 当前={total_exposure:.2f}, "
                        f"新增={signal.volume:.2f}, "
                        f"最大={self.max_position_size:.2f}"
                    )
                    return False
            
            # 2. 检查可用保证金
            snapshot = self.account_manager.get_account_snapshot()
            if snapshot:
                # 简单估算所需保证金（实际应该更精确）
                required_margin = signal.volume * signal.price * 0.1 if signal.price > 0 else signal.volume * 100000 * 0.1
                if snapshot.margin_free < required_margin:
                    logger.warning(
                        f"保证金不足: 可用={snapshot.margin_free:.2f}, "
                        f"需要={required_margin:.2f}"
                    )
                    return False
            
            # 3. 检查信号合理性
            if signal.volume <= 0:
                logger.warning(f"交易量无效: {signal.volume}")
                return False
            
            if signal.action == 'CLOSE' and not current_position:
                logger.warning("平仓信号但没有持仓")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"风控检查失败: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            统计字典
        """
        return {
            **self.stats,
            'account_snapshot': self.account_manager.get_account_snapshot().__dict__ if self.account_manager.get_account_snapshot() else None,
        }

