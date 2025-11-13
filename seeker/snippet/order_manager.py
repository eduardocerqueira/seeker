#date: 2025-11-13T17:02:59Z
#url: https://api.github.com/gists/c1c2c32c02ad6b73f59da7fc63add2f0
#owner: https://api.github.com/users/wangwei334455

"""
订单管理器 (OrderManager)
负责监听异步订单回报，维护订单状态，并通知 AccountStateManager 更新账户

【核心职责】
1. 追踪所有待处理的订单（Pending Orders）
2. 监听异步执行报告（ExecutionReport）
3. 处理订单状态转换（NEW → OPEN → FILLED）
4. 通知 AccountStateManager 更新持仓和账户状态

【设计原则】
- 单一数据源：只相信交易所的回报，不相信发出的请求
- 状态一致性：确保订单状态和账户状态同步
- 异步处理：非阻塞处理执行报告
"""
from typing import Dict, Optional, Any
from loguru import logger
import time

from src.trading.types.execution_report import ExecutionReport, PendingOrder
from src.trading.services.account_state_manager import AccountStateManager


class OrderManager:
    """
    订单管理器
    
    【工作流程】
    1. TradeExecutorService 发送订单后，调用 add_pending_order() 加入追踪
    2. 交易所异步推送 ExecutionReport
    3. OrderManager.process_report() 处理报告
    4. 通知 AccountStateManager 更新持仓
    5. 订单完成后从追踪列表移除
    """
    
    def __init__(self, account_manager: AccountStateManager):
        """
        初始化订单管理器
        
        Args:
            account_manager: 账户状态管理器，用于更新持仓
        """
        self.account_manager = account_manager
        
        # 内部字典，用于追踪所有未完成的订单: {order_id: PendingOrder}
        self.pending_orders: Dict[int, PendingOrder] = {}
        
        # 统计信息
        self.stats = {
            'total_orders': 0,
            'filled_orders': 0,
            'canceled_orders': 0,
            'rejected_orders': 0,
            'partial_fills': 0,
        }
        
        logger.info("⚡ 订单管理器 (OrderManager) 已启动")
    
    def add_pending_order(
        self,
        order_id: int,
        symbol: str,
        action: str,
        requested_volume: float,
        magic: int,
        timestamp: Optional[int] = None,
    ):
        """
        由 TradeExecutorService 调用：将成功发送的订单请求加入追踪列表
        
        Args:
            order_id: 订单ID（来自交易所）
            symbol: 交易品种
            action: 交易动作（'BUY', 'SELL', 'CLOSE'）
            requested_volume: 请求的交易量
            magic: 魔术号
            timestamp: 创建时间戳（如果为None则使用当前时间）
        """
        pending_order = PendingOrder(
            order_id=order_id,
            symbol=symbol,
            action=action,
            requested_volume=requested_volume,
            creation_time=timestamp or int(time.time() * 1000),
            magic=magic,
        )
        
        self.pending_orders[order_id] = pending_order
        self.stats['total_orders'] += 1
        
        logger.info(
            f"➕ 订单 {order_id} ({action} {requested_volume}手) 已加入待处理列表 | "
            f"Magic={magic}"
        )
    
    def process_report(self, report: ExecutionReport):
        """
        处理来自交易所的异步执行报告
        
        这是核心逻辑，触发账户状态的改变
        
        Args:
            report: ExecutionReport 实例
        """
        order_id = report.order_id
        
        if order_id not in self.pending_orders:
            # 可能是系统重启后收到的旧回报，或不是我们系统发出的订单
            logger.warning(f"接收到未知订单ID ({order_id}) 的回报，忽略")
            return
        
        pending_order = self.pending_orders[order_id]
        
        logger.info(
            f"🔄 处理订单 {order_id} 回报: "
            f"ExecType={report.exec_type}, Status={report.order_status}"
        )
        
        # 1. 关键逻辑：处理成交 (FILL / PARTIAL_FILL)
        if report.exec_type in ['FILL', 'PARTIAL_FILL']:
            # 更新内部追踪状态
            pending_order.cumulative_filled_volume = report.cumulative_volume
            
            if report.exec_type == 'PARTIAL_FILL':
                pending_order.current_status = 'OPEN'
                self.stats['partial_fills'] += 1
            else:
                pending_order.current_status = 'FILLED'
                self.stats['filled_orders'] += 1
            
            # **通知 AccountStateManager 更新持仓和账户**
            # 注意：对于 CLOSE 操作，需要特殊处理
            if pending_order.action == 'CLOSE':
                # 平仓操作：减少持仓量
                self.account_manager.close_position_from_execution(
                    symbol=report.symbol,
                    volume_closed=report.last_fill_volume,
                    close_price=report.last_fill_price,
                    order_id=report.order_id,
                    magic=report.magic,
                )
            else:
                # 开仓操作：增加持仓量
                self.account_manager.update_position_from_execution(
                    symbol=report.symbol,
                    side=report.side,
                    volume_change=report.last_fill_volume,
                    fill_price=report.last_fill_price,
                    order_id=report.order_id,
                    magic=report.magic,
                )
            
            logger.success(
                f"💰 成功成交! 订单 {order_id} 填充 {report.last_fill_volume}手 @ {report.last_fill_price:.2f} | "
                f"累计成交: {report.cumulative_volume}手"
            )
        
        # 2. 处理订单取消
        elif report.exec_type == 'CANCEL':
            pending_order.current_status = 'CANCELED'
            self.stats['canceled_orders'] += 1
            logger.warning(f"🗑️ 订单 {order_id} 被取消: {report.comment}")
        
        # 3. 处理订单拒绝
        elif report.exec_type == 'REJECT':
            pending_order.current_status = 'REJECTED'
            self.stats['rejected_orders'] += 1
            logger.error(f"❌ 订单 {order_id} 被拒绝: {report.comment}")
        
        # 4. 处理订单新建（NEW）
        elif report.exec_type == 'NEW':
            pending_order.current_status = 'OPEN'
            logger.debug(f"📝 订单 {order_id} 已进入交易所系统")
        
        # 5. 最终状态处理：订单完成或取消，从追踪列表移除
        if report.order_status in ['FILLED', 'CANCELED', 'REJECTED']:
            if pending_order.current_status != 'FILLED' and report.order_status == 'FILLED':
                pending_order.current_status = 'FILLED'
                self.stats['filled_orders'] += 1
            
            if pending_order.current_status == 'FILLED':
                logger.success(f"✅ 订单 {order_id} 已完全成交，从追踪列表移除")
            elif report.order_status == 'CANCELED':
                logger.warning(f"🗑️ 订单 {order_id} 被取消，从追踪列表移除")
            elif report.order_status == 'REJECTED':
                logger.error(f"❌ 订单 {order_id} 被拒绝，从追踪列表移除")
            
            # 从待处理列表中移除
            del self.pending_orders[order_id]
    
    def get_pending_orders(self) -> Dict[int, PendingOrder]:
        """
        获取当前所有待处理订单
        
        Returns:
            待处理订单字典
        """
        return self.pending_orders.copy()
    
    def get_pending_order(self, order_id: int) -> Optional[PendingOrder]:
        """
        获取指定订单的待处理状态
        
        Args:
            order_id: 订单ID
            
        Returns:
            PendingOrder 或 None
        """
        return self.pending_orders.get(order_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            统计字典
        """
        return {
            **self.stats,
            'pending_count': len(self.pending_orders),
        }

