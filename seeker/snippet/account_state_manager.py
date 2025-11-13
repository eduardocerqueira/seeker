#date: 2025-11-13T17:02:59Z
#url: https://api.github.com/gists/c1c2c32c02ad6b73f59da7fc63add2f0
#owner: https://api.github.com/users/wangwei334455

"""
账户状态管理器 (Account State Manager)
负责维护实时账户状态，包括账户信息、持仓和挂单

【核心职责】
1. 消费 ACCOUNT_UPDATE 消息（从 WebSocket 或 Redis PubSub）
2. 维护账户状态（净值、保证金、持仓、挂单）
3. 提供 O(1) 复杂度的查询接口

【设计原则】
- 线程安全：使用锁保护共享状态
- 原子性更新：ACCOUNT_UPDATE 消息包含完整快照，确保状态一致性
- 高效查询：使用字典结构实现 O(1) 查询
"""
import json
import time
import threading
from typing import Dict, List, Optional, Set
from collections import defaultdict
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class AccountSnapshot:
    """账户快照数据"""
    account_id: str
    equity: float = 0.0              # 净值
    balance: float = 0.0              # 余额
    margin_free: float = 0.0          # 可用保证金
    margin: float = 0.0                # 已用保证金
    margin_level: float = 0.0          # 保证金水平 (%)
    total_floating_pnl: float = 0.0    # 总浮动盈亏
    currency: str = 'USD'              # 账户币种
    timestamp: int = 0                 # 时间戳（毫秒）


@dataclass
class PositionData:
    """持仓数据"""
    ticket: int                        # 持仓ID / 订单号
    symbol: str                        # 交易品种
    side: str                          # 'BUY' 或 'SELL'
    volume: float                      # 持仓量 (手)
    open_price: float                  # 开仓价格
    current_price: float               # 当前价格
    floating_pnl: float                # 浮动盈亏
    swap: float = 0.0                  # 隔夜利息/掉期
    time: int = 0                      # 开仓时间 (毫秒)
    magic: int = 0                     # MT5 魔术号
    stop_loss: Optional[float] = None  # 止损价格
    take_profit: Optional[float] = None # 止盈价格
    
    @property
    def position_id(self) -> str:
        """返回持仓ID（字符串格式）"""
        return str(self.ticket)


@dataclass
class OrderData:
    """挂单数据"""
    ticket: int                        # 订单ID
    symbol: str                        # 交易品种
    type: str                          # 订单类型 ('BUY_LIMIT', 'SELL_LIMIT', 'BUY_STOP', 'SELL_STOP')
    volume: float                      # 订单量 (手)
    price: float                       # 委托价格
    time: int = 0                      # 下单时间 (毫秒)
    magic: int = 0                     # MT5 魔术号
    
    @property
    def order_id(self) -> str:
        """返回订单ID（字符串格式）"""
        return str(self.ticket)


class AccountStateManager:
    """
    账户状态管理器
    
    【状态维护】
    - 账户快照：净值、保证金等
    - 持仓字典：{position_id: PositionData}
    - 挂单字典：{order_id: OrderData}
    
    【查询接口】
    - is_position_open(symbol, magic) -> bool
    - get_position(symbol, magic) -> Optional[PositionData]
    - get_total_exposure() -> float
    - get_positions_by_symbol(symbol) -> List[PositionData]
    - get_positions_by_magic(magic) -> List[PositionData]
    """
    
    def __init__(self, account_id: str = 'default'):
        """
        初始化账户状态管理器
        
        Args:
            account_id: 账户ID
        """
        self.account_id = account_id
        
        # 账户快照
        self.snapshot: Optional[AccountSnapshot] = None
        
        # 持仓字典：{position_id: PositionData}
        self.positions: Dict[str, PositionData] = {}
        
        # 挂单字典：{order_id: OrderData}
        self.orders: Dict[str, OrderData] = {}
        
        # 索引：按 symbol 索引持仓
        self.positions_by_symbol: Dict[str, Set[str]] = defaultdict(set)
        
        # 索引：按 magic 索引持仓
        self.positions_by_magic: Dict[int, Set[str]] = defaultdict(set)
        
        # 索引：按 symbol 索引挂单
        self.orders_by_symbol: Dict[str, Set[str]] = defaultdict(set)
        
        # 索引：按 magic 索引挂单
        self.orders_by_magic: Dict[int, Set[str]] = defaultdict(set)
        
        # 线程锁
        self.lock = threading.RLock()
        
        # 统计信息
        self.stats = {
            'update_count': 0,
            'last_update_time': 0,
        }
        
        logger.info(f"账户状态管理器已初始化: account_id={account_id}")
    
    def update_from_account_update(self, payload: Dict) -> None:
        """
        从 ACCOUNT_UPDATE 消息更新账户状态
        
        【消息格式】
        {
            'snapshot': {
                'accountId': str,
                'equity': float,
                'balance': float,
                'marginFree': float,
                'margin': float,
                'marginLevel': float,
                'totalFloatingPnL': float,
                'currency': str,
            },
            'positions': [
                {
                    'ticket': int,
                    'symbol': str,
                    'type': 'BUY' | 'SELL',
                    'volume': float,
                    'openPrice': float,
                    'currentPrice': float,
                    'floatingPnL': float,
                    'swap': float,
                    'time': int,
                    'magic': int,
                    'stopLoss': float (optional),
                    'takeProfit': float (optional),
                },
                ...
            ],
            'orders': [
                {
                    'ticket': int,
                    'symbol': str,
                    'type': str,
                    'volume': float,
                    'price': float,
                    'time': int,
                    'magic': int,
                },
                ...
            ],
        }
        
        Args:
            payload: ACCOUNT_UPDATE 消息体
        """
        with self.lock:
            try:
                # 1. 更新账户快照
                snapshot_data = payload.get('snapshot', {})
                self.snapshot = AccountSnapshot(
                    account_id=snapshot_data.get('accountId', self.account_id),
                    equity=float(snapshot_data.get('equity', 0.0)),
                    balance=float(snapshot_data.get('balance', 0.0)),
                    margin_free=float(snapshot_data.get('marginFree', 0.0)),
                    margin=float(snapshot_data.get('margin', 0.0)),
                    margin_level=float(snapshot_data.get('marginLevel', 0.0)),
                    total_floating_pnl=float(snapshot_data.get('totalFloatingPnL', 0.0)),
                    currency=snapshot_data.get('currency', 'USD'),
                    timestamp=int(time.time() * 1000),
                )
                
                # 2. 清空旧索引
                self.positions_by_symbol.clear()
                self.positions_by_magic.clear()
                self.orders_by_symbol.clear()
                self.orders_by_magic.clear()
                
                # 3. 更新持仓（原子性替换）
                positions_data = payload.get('positions', [])
                new_positions: Dict[str, PositionData] = {}
                
                for pos_data in positions_data:
                    position = PositionData(
                        ticket=int(pos_data.get('ticket', 0)),
                        symbol=pos_data.get('symbol', ''),
                        side=pos_data.get('type', 'BUY'),  # 'BUY' 或 'SELL'
                        volume=float(pos_data.get('volume', 0.0)),
                        open_price=float(pos_data.get('openPrice', 0.0)),
                        current_price=float(pos_data.get('currentPrice', 0.0)),
                        floating_pnl=float(pos_data.get('floatingPnL', 0.0)),
                        swap=float(pos_data.get('swap', 0.0)),
                        time=int(pos_data.get('time', 0)),
                        magic=int(pos_data.get('magic', 0)),
                        stop_loss=pos_data.get('stopLoss'),
                        take_profit=pos_data.get('takeProfit'),
                    )
                    
                    position_id = position.position_id
                    new_positions[position_id] = position
                    
                    # 更新索引
                    self.positions_by_symbol[position.symbol].add(position_id)
                    self.positions_by_magic[position.magic].add(position_id)
                
                self.positions = new_positions
                
                # 4. 更新挂单（原子性替换）
                orders_data = payload.get('orders', [])
                new_orders: Dict[str, OrderData] = {}
                
                for order_data in orders_data:
                    order = OrderData(
                        ticket=int(order_data.get('ticket', 0)),
                        symbol=order_data.get('symbol', ''),
                        type=order_data.get('type', ''),
                        volume=float(order_data.get('volume', 0.0)),
                        price=float(order_data.get('price', 0.0)),
                        time=int(order_data.get('time', 0)),
                        magic=int(order_data.get('magic', 0)),
                    )
                    
                    order_id = order.order_id
                    new_orders[order_id] = order
                    
                    # 更新索引
                    self.orders_by_symbol[order.symbol].add(order_id)
                    self.orders_by_magic[order.magic].add(order_id)
                
                self.orders = new_orders
                
                # 5. 更新统计
                self.stats['update_count'] += 1
                self.stats['last_update_time'] = int(time.time() * 1000)
                
                logger.debug(
                    f"账户状态已更新: "
                    f"持仓={len(self.positions)}, "
                    f"挂单={len(self.orders)}, "
                    f"净值={self.snapshot.equity:.2f}"
                )
                
            except Exception as e:
                logger.error(f"更新账户状态失败: {e}")
                raise
    
    # ==================== 查询接口 ====================
    
    def is_position_open(self, symbol: str, magic: Optional[int] = None) -> bool:
        """
        检查是否有持仓
        
        Args:
            symbol: 交易品种
            magic: 魔术号（可选，如果提供则只检查该魔术号的持仓）
            
        Returns:
            bool: 是否有持仓
        """
        with self.lock:
            if magic is not None:
                # 检查特定魔术号的持仓
                position_ids = self.positions_by_magic.get(magic, set())
                for position_id in position_ids:
                    position = self.positions.get(position_id)
                    if position and position.symbol == symbol:
                        return True
                return False
            else:
                # 检查所有持仓
                position_ids = self.positions_by_symbol.get(symbol, set())
                return len(position_ids) > 0
    
    def get_position(self, symbol: str, magic: Optional[int] = None) -> Optional[PositionData]:
        """
        获取持仓（如果有多笔持仓，返回第一笔）
        
        Args:
            symbol: 交易品种
            magic: 魔术号（可选）
            
        Returns:
            PositionData 或 None
        """
        with self.lock:
            if magic is not None:
                position_ids = self.positions_by_magic.get(magic, set())
                for position_id in position_ids:
                    position = self.positions.get(position_id)
                    if position and position.symbol == symbol:
                        return position
            else:
                position_ids = self.positions_by_symbol.get(symbol, set())
                if position_ids:
                    position_id = next(iter(position_ids))
                    return self.positions.get(position_id)
            return None
    
    def get_positions_by_symbol(self, symbol: str) -> List[PositionData]:
        """
        获取指定品种的所有持仓
        
        Args:
            symbol: 交易品种
            
        Returns:
            持仓列表
        """
        with self.lock:
            position_ids = self.positions_by_symbol.get(symbol, set())
            return [
                self.positions[position_id]
                for position_id in position_ids
                if position_id in self.positions
            ]
    
    def get_positions_by_magic(self, magic: int) -> List[PositionData]:
        """
        获取指定魔术号的所有持仓
        
        Args:
            magic: 魔术号
            
        Returns:
            持仓列表
        """
        with self.lock:
            position_ids = self.positions_by_magic.get(magic, set())
            return [
                self.positions[position_id]
                for position_id in position_ids
                if position_id in self.positions
            ]
    
    def get_all_positions(self) -> List[PositionData]:
        """
        获取所有持仓
        
        Returns:
            持仓列表
        """
        with self.lock:
            return list(self.positions.values())
    
    def get_total_exposure(self, symbol: Optional[str] = None) -> float:
        """
        获取总持仓量（总敞口）
        
        Args:
            symbol: 交易品种（可选，如果提供则只计算该品种）
            
        Returns:
            总持仓量（手）
        """
        with self.lock:
            if symbol:
                positions = self.get_positions_by_symbol(symbol)
            else:
                positions = self.get_all_positions()
            
            return sum(pos.volume for pos in positions)
    
    def get_net_position(self, symbol: str, magic: Optional[int] = None) -> float:
        """
        获取净持仓量（多头 - 空头）
        
        Args:
            symbol: 交易品种
            magic: 魔术号（可选）
            
        Returns:
            净持仓量（正数=净多头，负数=净空头）
        """
        with self.lock:
            if magic is not None:
                positions = self.get_positions_by_magic(magic)
                positions = [p for p in positions if p.symbol == symbol]
            else:
                positions = self.get_positions_by_symbol(symbol)
            
            net = 0.0
            for pos in positions:
                if pos.side == 'BUY':
                    net += pos.volume
                else:  # SELL
                    net -= pos.volume
            
            return net
    
    def get_total_floating_pnl(self, symbol: Optional[str] = None, magic: Optional[int] = None) -> float:
        """
        获取总浮动盈亏
        
        Args:
            symbol: 交易品种（可选）
            magic: 魔术号（可选）
            
        Returns:
            总浮动盈亏
        """
        with self.lock:
            if magic is not None:
                positions = self.get_positions_by_magic(magic)
                if symbol:
                    positions = [p for p in positions if p.symbol == symbol]
            elif symbol:
                positions = self.get_positions_by_symbol(symbol)
            else:
                positions = self.get_all_positions()
            
            return sum(pos.floating_pnl for pos in positions)
    
    def has_open_order(self, symbol: str, magic: Optional[int] = None) -> bool:
        """
        检查是否有挂单
        
        Args:
            symbol: 交易品种
            magic: 魔术号（可选）
            
        Returns:
            bool: 是否有挂单
        """
        with self.lock:
            if magic is not None:
                order_ids = self.orders_by_magic.get(magic, set())
                for order_id in order_ids:
                    order = self.orders.get(order_id)
                    if order and order.symbol == symbol:
                        return True
                return False
            else:
                order_ids = self.orders_by_symbol.get(symbol, set())
                return len(order_ids) > 0
    
    def get_orders_by_symbol(self, symbol: str) -> List[OrderData]:
        """
        获取指定品种的所有挂单
        
        Args:
            symbol: 交易品种
            
        Returns:
            挂单列表
        """
        with self.lock:
            order_ids = self.orders_by_symbol.get(symbol, set())
            return [
                self.orders[order_id]
                for order_id in order_ids
                if order_id in self.orders
            ]
    
    def get_orders_by_magic(self, magic: int) -> List[OrderData]:
        """
        获取指定魔术号的所有挂单
        
        Args:
            magic: 魔术号
            
        Returns:
            挂单列表
        """
        with self.lock:
            order_ids = self.orders_by_magic.get(magic, set())
            return [
                self.orders[order_id]
                for order_id in order_ids
                if order_id in self.orders
            ]
    
    def get_account_snapshot(self) -> Optional[AccountSnapshot]:
        """
        获取账户快照
        
        Returns:
            AccountSnapshot 或 None
        """
        with self.lock:
            return self.snapshot
    
    def _get_position_key(self, symbol: str, magic: int) -> str:
        """
        生成持仓唯一键
        
        Args:
            symbol: 交易品种
            magic: 魔术号
            
        Returns:
            唯一键字符串: "{symbol}_{magic}"
        """
        return f"{symbol}_{magic}"
    
    def update_position_from_execution(
        self,
        symbol: str,
        side: str,
        volume_change: float,
        fill_price: float,
        order_id: int,
        magic: int,
    ) -> None:
        """
        根据异步成交回报（FILL Report）更新账户持仓
        
        【核心逻辑】
        - 使用 symbol_magic 作为唯一键
        - 处理开仓、加仓、平仓、反手等所有场景
        - 更新保证金和净值
        
        【场景处理】
        1. 开仓：当前无持仓，创建新持仓
        2. 加仓：持仓方向与成交方向一致，增加持仓量（加权平均开仓价）
        3. 平仓：持仓方向与成交方向相反，减少持仓量
        4. 反手：平仓后仍有剩余成交量，反向开仓
        
        Args:
            symbol: 交易品种
            side: 交易方向 ('BUY' 或 'SELL')
            volume_change: 本次成交的数量（手）
            fill_price: 本次成交的实际价格
            order_id: 订单ID
            magic: 策略魔术号
        """
        with self.lock:
            try:
                # 生成唯一键
                position_key = self._get_position_key(symbol, magic)
                
                # 确定本次操作方向（0=BUY, 1=SELL）
                action_type = 0 if side == 'BUY' else 1
                
                # 1. 尝试获取现有持仓
                current_position = None
                position_id = None
                
                # 查找同品种、同魔术号的持仓
                for pid in self.positions_by_symbol.get(symbol, set()):
                    pos = self.positions.get(pid)
                    if pos and pos.symbol == symbol and pos.magic == magic:
                        current_position = pos
                        position_id = pid
                        break
                
                if not current_position:
                    # --- 情景 A: 开仓（当前无持仓） ---
                    new_position = PositionData(
                        ticket=order_id,
                        symbol=symbol,
                        side=side,
                        volume=volume_change,
                        open_price=fill_price,
                        current_price=fill_price,
                        floating_pnl=0.0,  # 刚开仓，浮盈为0
                        swap=0.0,
                        time=int(time.time() * 1000),
                        magic=magic,
                    )
                    
                    position_id = new_position.position_id
                    self.positions[position_id] = new_position
                    
                    # 更新索引
                    self.positions_by_symbol[symbol].add(position_id)
                    self.positions_by_magic[magic].add(position_id)
                    
                    logger.success(
                        f"📈 新持仓创建: {side} {volume_change:.2f}手 @ {fill_price:.2f} | "
                        f"键: {position_key}, 持仓ID: {position_id}"
                    )
                
                else:
                    # 获取持仓方向（0=BUY, 1=SELL）
                    position_type = 0 if current_position.side == 'BUY' else 1
                    
                    if position_type == action_type:
                        # --- 情景 B: 加仓（持仓方向与成交方向一致） ---
                        # 采用加权平均法计算新的开仓价格
                        old_volume = current_position.volume
                        old_price = current_position.open_price
                        total_volume = old_volume + volume_change
                        new_open_price = (old_volume * old_price + volume_change * fill_price) / total_volume
                        
                        current_position.volume = total_volume
                        current_position.open_price = new_open_price
                        current_position.current_price = fill_price
                        
                        logger.info(
                            f"➡️ 持仓加仓: {side} {volume_change:.2f}手 @ {fill_price:.2f} | "
                            f"原持仓={old_volume:.2f}手@{old_price:.2f} | "
                            f"新均价={new_open_price:.2f} | 新持仓={total_volume:.2f}手"
                        )
                    
                    else:
                        # --- 情景 C: 平仓（持仓方向与成交方向相反） ---
                        remaining_volume = current_position.volume - volume_change
                        
                        # 计算平仓盈亏
                        entry_price = current_position.open_price
                        direction = 1 if position_type == 0 else -1  # 1=多头, -1=空头
                        pnl_per_lot = (fill_price - entry_price) * direction
                        closed_pnl = pnl_per_lot * min(volume_change, current_position.volume)
                        
                        if remaining_volume > 0:
                            # 1. 部分平仓
                            current_position.volume = remaining_volume
                            current_position.current_price = fill_price
                            
                            logger.warning(
                                f"⬅️ 持仓部分平仓: {volume_change:.2f}手 @ {fill_price:.2f} | "
                                f"原持仓={current_position.volume + volume_change:.2f}手@{entry_price:.2f} | "
                                f"剩余={remaining_volume:.2f}手 | 本次盈亏={closed_pnl:.2f}"
                            )
                        
                        else:
                            # 2. 完全平仓或反手开仓
                            old_volume = current_position.volume
                            
                            # 删除原持仓
                            del self.positions[position_id]
                            self.positions_by_symbol[symbol].discard(position_id)
                            self.positions_by_magic[magic].discard(position_id)
                            
                            # 计算完全平仓的盈亏
                            full_closed_pnl = pnl_per_lot * old_volume
                            
                            logger.success(
                                f"❌ 持仓完全平仓: {old_volume:.2f}手 @ {fill_price:.2f} | "
                                f"开仓价={entry_price:.2f} | 盈亏={full_closed_pnl:.2f}"
                            )
                            
                            # 更新账户余额和净值（平仓盈亏）
                            if self.snapshot:
                                self.snapshot.balance += full_closed_pnl
                                self.snapshot.equity = self.snapshot.balance
                            
                            if remaining_volume < 0:
                                # 3. 反手开仓：如果成交量大于原有持仓量，剩余部分为反手的新开仓
                                new_side = 'BUY' if position_type == 1 else 'SELL'
                                new_volume = abs(remaining_volume)
                                
                                logger.info(
                                    f"🔄 反手开仓: {new_side} {new_volume:.2f}手 @ {fill_price:.2f}"
                                )
                                
                                # 递归调用自己，以新的空仓状态处理剩余的成交量
                                self.update_position_from_execution(
                                    symbol=symbol,
                                    side=new_side,
                                    volume_change=new_volume,
                                    fill_price=fill_price,
                                    order_id=order_id,
                                    magic=magic,
                                )
                
                # 3. 更新账户的保证金占用和可用保证金
                self._update_margin_and_equity_status()
                
            except Exception as e:
                logger.error(f"从执行报告更新持仓失败: {e}")
                raise
    
    def close_position_from_execution(
        self,
        symbol: str,
        volume_closed: float,
        close_price: float,
        order_id: int,
        magic: int,
    ) -> None:
        """
        从执行报告平仓（CLOSE 操作专用）
        
        【说明】
        - 这是专门处理 CLOSE 信号的方法
        - 与 update_position_from_execution 的区别：CLOSE 操作明确指定要平仓
        - 如果持仓方向不明确，使用 update_position_from_execution 更合适
        
        Args:
            symbol: 交易品种
            volume_closed: 平仓数量（手）
            close_price: 平仓价格
            order_id: 订单ID
            magic: 魔术号
        """
        with self.lock:
            try:
                # 查找持仓（同品种、同魔术号）
                position_to_close = None
                position_id_to_close = None
                
                for position_id in self.positions_by_symbol.get(symbol, set()):
                    position = self.positions.get(position_id)
                    if position and position.symbol == symbol and position.magic == magic:
                        position_to_close = position
                        position_id_to_close = position_id
                        break
                
                if not position_to_close:
                    logger.warning(f"平仓失败: 未找到持仓 {symbol} Magic={magic}")
                    return
                
                # 计算盈亏
                entry_price = position_to_close.open_price
                position_type = 0 if position_to_close.side == 'BUY' else 1
                direction = 1 if position_type == 0 else -1
                pnl_per_lot = (close_price - entry_price) * direction
                total_pnl = pnl_per_lot * min(volume_closed, position_to_close.volume)
                
                # 减少持仓量
                old_volume = position_to_close.volume
                new_volume = old_volume - volume_closed
                
                if new_volume <= 0:
                    # 完全平仓：删除持仓
                    del self.positions[position_id_to_close]
                    self.positions_by_symbol[symbol].discard(position_id_to_close)
                    self.positions_by_magic[magic].discard(position_id_to_close)
                    
                    # 更新账户余额和净值
                    if self.snapshot:
                        self.snapshot.balance += total_pnl
                        self.snapshot.equity = self.snapshot.balance
                    
                    logger.info(
                        f"持仓已完全平仓: {symbol} | "
                        f"原持仓={old_volume:.2f}手@{entry_price:.2f} | "
                        f"平仓={volume_closed:.2f}手@{close_price:.2f} | "
                        f"盈亏={total_pnl:.2f}"
                    )
                else:
                    # 部分平仓：更新持仓量
                    position_to_close.volume = new_volume
                    position_to_close.current_price = close_price
                    # 重新计算浮盈（基于剩余持仓）
                    position_to_close.floating_pnl = pnl_per_lot * new_volume
                    
                    logger.info(
                        f"持仓已部分平仓: {symbol} | "
                        f"原持仓={old_volume:.2f}手@{entry_price:.2f} | "
                        f"平仓={volume_closed:.2f}手@{close_price:.2f} | "
                        f"剩余={new_volume:.2f}手 | "
                        f"本次盈亏={total_pnl:.2f}"
                    )
                
                # 更新保证金和净值
                self._update_margin_and_equity_status()
                
            except Exception as e:
                logger.error(f"从执行报告平仓失败: {e}")
                raise
    
    def _update_margin_and_equity_status(self, current_market_price: Optional[float] = None):
        """
        更新浮动盈亏、净值和可用保证金
        
        【说明】
        - 这是一个简化版本，实际系统中需要接入实时行情数据
        - 假设 10倍杠杆（即 10% 保证金比例）
        
        Args:
            current_market_price: 当前市场价格（如果为None，使用持仓的current_price）
        """
        with self.lock:
            if not self.snapshot:
                return
            
            total_pnl = 0.0
            total_margin_used = 0.0
            
            # 计算所有持仓的浮动盈亏和保证金占用
            for pos in self.positions.values():
                # 获取当前价格（优先使用传入的价格，否则使用持仓的current_price）
                market_price = current_market_price or pos.current_price
                
                # 浮动盈亏计算
                direction = 1 if pos.side == 'BUY' else -1
                pnl_per_unit = (market_price - pos.open_price) * direction
                pos.floating_pnl = pnl_per_unit * pos.volume
                total_pnl += pos.floating_pnl
                
                # 保证金占用计算（简化：假设 10倍杠杆，即 10% 保证金）
                # 实际应该从账户信息获取杠杆比例
                margin_ratio = 0.1  # 10% 保证金
                total_margin_used += pos.volume * market_price * margin_ratio
                
                # 更新持仓的当前价格
                pos.current_price = market_price
            
            # 更新账户快照
            # 净值 = 余额 + 浮动盈亏
            self.snapshot.equity = self.snapshot.balance + total_pnl
            # 可用保证金 = 净值 - 已用保证金
            self.snapshot.margin_free = self.snapshot.equity - total_margin_used
            
            # 更新总浮动盈亏
            self.snapshot.total_floating_pnl = total_pnl
            
            # 计算保证金水平（百分比）
            if total_margin_used > 0:
                self.snapshot.margin_level = (self.snapshot.equity / total_margin_used) * 100
            else:
                self.snapshot.margin_level = 0.0
            
            # 风险提示：保证金不足
            if self.snapshot.margin_free < 0:
                logger.critical(
                    f"🚨 保证金不足！Margin Call 风险！"
                    f"可用保证金: {self.snapshot.margin_free:.2f} | "
                    f"净值: {self.snapshot.equity:.2f} | "
                    f"已用保证金: {total_margin_used:.2f}"
                )
    
    def get_stats(self) -> Dict:
        """
        获取统计信息
        
        Returns:
            统计字典
        """
        with self.lock:
            return {
                **self.stats,
                'positions_count': len(self.positions),
                'orders_count': len(self.orders),
                'snapshot': self.snapshot.__dict__ if self.snapshot else None,
            }

