#date: 2025-11-13T17:02:59Z
#url: https://api.github.com/gists/c1c2c32c02ad6b73f59da7fc63add2f0
#owner: https://api.github.com/users/wangwei334455

"""
统一交易执行服务 (Unified Trade Executor Service)
总指挥官：整合所有组件，驱动完整的交易流程

【核心职责】
1. 监听特征数据（来自 FeatureEngine）
2. 监听账户更新（来自 AccountStateManager）
3. 驱动决策流程：特征 → 信号生成 → 交易执行
4. 协调所有子系统

【完整流程】
on_new_bar/on_feature_update → SignalGenerator → TradeExecutorAdapter → GrpcTradeClient

【设计原则】
- 单一入口：所有交易决策都通过这个服务
- 事件驱动：基于特征更新和账户更新触发决策
- 解耦设计：各组件独立，通过接口交互
"""
import json
import time
import threading
from typing import Dict, Any, Optional, Callable
from queue import Queue
from loguru import logger

from src.trading.services.account_state_manager import AccountStateManager
from src.trading.services.signal_generator import SignalGenerator, TradingSignal
from src.trading.services.trade_executor_adapter import TradeExecutorAdapter
from src.trading.services.grpc_trade_client import get_grpc_client


class UnifiedTradeExecutor:
    """
    统一交易执行服务（总指挥官）
    
    【组件整合】
    - AccountStateManager: 管理账户状态
    - SignalGenerator: 生成交易信号
    - TradeExecutorAdapter: 执行交易信号
    - GrpcTradeClient: gRPC 通信
    
    【事件驱动】
    - on_feature_update: 特征数据更新时触发
    - on_account_update: 账户状态更新时触发
    - on_new_bar: 新K线收盘时触发（可选）
    """
    
    def __init__(
        self,
        symbol: str,
        account_id: str = 'default',
        default_magic: int = 202409,
        default_volume: float = 0.01,
        max_position_size: float = 1.0,
        grpc_client=None,
    ):
        """
        初始化统一交易执行服务
        
        Args:
            symbol: 交易品种
            account_id: 账户ID
            default_magic: 默认魔术号
            default_volume: 默认交易量（手）
            max_position_size: 最大持仓量（手）
            grpc_client: gRPC客户端（如果为None则使用单例）
        """
        self.symbol = symbol
        self.account_id = account_id
        
        # 初始化核心组件
        self.account_manager = AccountStateManager(account_id=account_id)
        self.signal_generator = SignalGenerator(
            account_manager=self.account_manager,
            default_magic=default_magic,
            default_volume=default_volume,
            max_position_size=max_position_size,
        )
        self.executor_adapter = TradeExecutorAdapter(
            grpc_client=grpc_client or get_grpc_client(),
            account_id=account_id,
        )
        
        # 事件队列（线程安全）
        self.feature_queue: Queue = Queue()
        self.account_queue: Queue = Queue()
        
        # 运行状态
        self.running = False
        self.worker_thread: Optional[threading.Thread] = None
        
        # 统计信息
        self.stats = {
            'feature_updates': 0,
            'account_updates': 0,
            'signals_generated': 0,
            'signals_executed': 0,
            'signals_failed': 0,
            'last_feature_time': 0,
            'last_account_time': 0,
        }
        
        logger.info(
            f"统一交易执行服务已初始化: "
            f"symbol={symbol}, account_id={account_id}, "
            f"magic={default_magic}, volume={default_volume}"
        )
    
    def start(self):
        """启动服务（启动后台工作线程）"""
        if self.running:
            logger.warning("服务已在运行")
            return
        
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        logger.info("统一交易执行服务已启动")
    
    def stop(self):
        """停止服务"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5.0)
        logger.info("统一交易执行服务已停止")
    
    def on_feature_update(self, features: Dict[str, Any]):
        """
        处理特征数据更新
        
        【触发时机】
        - Windows 端 FeatureEngine 计算完成
        - 通过 Redis PubSub 或 WebSocket 接收
        
        Args:
            features: 特征数据字典（来自 FeatureEngine）
        """
        try:
            # 验证特征数据
            if not features.get('symbol') or features.get('symbol') != self.symbol:
                return  # 忽略其他品种的特征
            
            # 放入队列（异步处理）
            self.feature_queue.put(('feature', features))
            self.stats['feature_updates'] += 1
            self.stats['last_feature_time'] = int(time.time() * 1000)
            
            logger.debug(f"特征数据已接收: {features.get('market_regime')}, RSI={features.get('rsi_14')}")
            
        except Exception as e:
            logger.error(f"处理特征更新失败: {e}")
    
    def on_account_update(self, account_update: Dict[str, Any]):
        """
        处理账户状态更新
        
        【触发时机】
        - MT5 Relay Service 推送账户更新
        - 通过 Redis PubSub 或 WebSocket 接收
        
        Args:
            account_update: 账户更新消息（ACCOUNT_UPDATE格式）
        """
        try:
            # 更新账户状态管理器（原子性更新）
            self.account_manager.update_from_account_update(account_update)
            self.stats['account_updates'] += 1
            self.stats['last_account_time'] = int(time.time() * 1000)
            
            logger.debug(
                f"账户状态已更新: "
                f"净值={self.account_manager.get_account_snapshot().equity if self.account_manager.get_account_snapshot() else 0:.2f}, "
                f"持仓数={len(self.account_manager.get_all_positions())}"
            )
            
            # 放入队列（触发决策流程）
            self.account_queue.put(('account', account_update))
            
        except Exception as e:
            logger.error(f"处理账户更新失败: {e}")
    
    def on_new_bar(self, kline: Dict[str, Any]):
        """
        处理新K线收盘事件（可选）
        
        【说明】
        - 如果 FeatureEngine 在 Windows 端，此方法可能不需要
        - 如果需要在 Linux 端计算特征，可以在这里触发
        
        Args:
            kline: K线数据
        """
        # 这里可以触发本地特征计算（如果需要）
        # 或者等待 Windows 端的特征数据
        logger.debug(f"新K线收盘: {kline.get('time')}")
    
    def _worker_loop(self):
        """
        后台工作线程：处理事件队列并驱动决策流程
        
        【处理流程】
        1. 从队列获取事件（特征更新或账户更新）
        2. 如果有特征数据，触发信号生成
        3. 如果有信号，执行交易
        """
        logger.info("工作线程已启动")
        
        while self.running:
            try:
                # 处理特征更新队列
                if not self.feature_queue.empty():
                    event_type, data = self.feature_queue.get_nowait()
                    if event_type == 'feature':
                        self._process_feature_update(data)
                
                # 处理账户更新队列（通常只更新状态，不触发决策）
                if not self.account_queue.empty():
                    event_type, data = self.account_queue.get_nowait()
                    # 账户更新已经在 on_account_update 中处理了
                    # 这里可以触发额外的逻辑（如持仓监控）
                
                # 短暂休眠，避免CPU占用过高
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"工作线程异常: {e}")
                time.sleep(1.0)
        
        logger.info("工作线程已停止")
    
    def _process_feature_update(self, features: Dict[str, Any]):
        """
        处理特征更新并生成交易信号
        
        【完整流程】
        1. 获取当前账户状态
        2. SignalGenerator 生成信号
        3. TradeExecutorAdapter 执行信号
        4. 更新统计信息
        
        Args:
            features: 特征数据
        """
        try:
            # 1. 生成交易信号
            signal = self.signal_generator.generate_signal(features)
            
            if signal:
                self.stats['signals_generated'] += 1
                logger.info(
                    f"🎯 交易信号已生成: {signal.action} {signal.symbol} "
                    f"{signal.volume}手 | 原因: {signal.reason} | "
                    f"置信度: {signal.confidence:.2f}"
                )
                
                # 2. 执行交易信号
                result = self.executor_adapter.execute_signal(signal)
                
                if result.get('success'):
                    self.stats['signals_executed'] += 1
                    logger.success(
                        f"✅ 交易执行成功: {signal.action} | "
                        f"订单号: {result.get('order_ticket', result.get('closed_count', 'N/A'))}"
                    )
                else:
                    self.stats['signals_failed'] += 1
                    logger.warning(
                        f"⚠️ 交易执行失败: {signal.action} | "
                        f"原因: {result.get('message', 'Unknown')}"
                    )
            else:
                logger.debug("未生成交易信号（策略条件不满足）")
                
        except Exception as e:
            logger.error(f"处理特征更新失败: {e}")
            self.stats['signals_failed'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            统计字典
        """
        return {
            **self.stats,
            'account_stats': self.account_manager.get_stats(),
            'signal_stats': self.signal_generator.get_stats(),
            'executor_stats': self.executor_adapter.get_stats(),
        }
    
    def get_account_state(self) -> Dict[str, Any]:
        """
        获取当前账户状态（用于监控和调试）
        
        Returns:
            账户状态字典
        """
        snapshot = self.account_manager.get_account_snapshot()
        positions = self.account_manager.get_all_positions()
        orders = self.account_manager.get_orders_by_symbol(self.symbol)
        
        return {
            'snapshot': snapshot.__dict__ if snapshot else None,
            'positions_count': len(positions),
            'positions': [pos.__dict__ for pos in positions],
            'orders_count': len(orders),
            'orders': [order.__dict__ for order in orders],
        }


# --- 使用示例 ---

def example_usage():
    """使用示例"""
    # 1. 初始化服务
    executor = UnifiedTradeExecutor(
        symbol='BTCUSDm',
        account_id='demo_12345',
        default_magic=202409,
        default_volume=0.01,
    )
    
    # 2. 启动服务
    executor.start()
    
    # 3. 模拟特征更新
    features = {
        'timestamp': int(time.time() * 1000),
        'symbol': 'BTCUSDm',
        'timeframe': '1m',
        'close_price': 105000.0,
        'market_regime': 'TREND_UP',
        'entry_signal': 'EXTREME_OVERSOLD',
        'rsi_14': 25.0,
        'atr_14': 100.0,
        'bb_upper': 105200.0,
        'bb_lower': 104800.0,
    }
    executor.on_feature_update(features)
    
    # 4. 模拟账户更新
    account_update = {
        'snapshot': {
            'accountId': 'demo_12345',
            'equity': 100000.0,
            'balance': 100000.0,
            'marginFree': 80000.0,
            'margin': 20000.0,
            'marginLevel': 500.0,
            'totalFloatingPnL': 0.0,
            'currency': 'USD',
        },
        'positions': [],
        'orders': [],
    }
    executor.on_account_update(account_update)
    
    # 5. 等待处理
    time.sleep(2.0)
    
    # 6. 查看统计
    stats = executor.get_stats()
    logger.info(f"统计信息: {stats}")
    
    # 7. 停止服务
    executor.stop()


if __name__ == '__main__':
    example_usage()

