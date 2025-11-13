#date: 2025-11-13T17:02:59Z
#url: https://api.github.com/gists/c1c2c32c02ad6b73f59da7fc63add2f0
#owner: https://api.github.com/users/wangwei334455

"""
交易执行服务：负责订单执行、订单-决策绑定和持久化记录

【职责】
1. 接收StrategyFSM生成的交易信号
2. 执行实时风控检查（Pre-Trade Risk Check）
3. 管理订单状态机（Order FSM）
4. 将决策ID绑定到交易所订单（ClientOrderID/Tag）
5. 处理交易所执行报告（成交、取消、拒绝）
6. 持久化订单-决策绑定记录（供前端可视化）
7. 连接管理和重试机制
"""
import json
import time
import redis
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from threading import Lock
from enum import Enum

logger = logging.getLogger(__name__)

# 导入风控管理器
try:
    from services.risk_controller import RiskController
except ImportError:
    RiskController = None
    logger.warning("RiskController未找到，风控功能将禁用")


# 订单状态定义（Order FSM）
class OrderStatus(Enum):
    """订单状态机状态定义"""
    NEW = 'NEW'              # 订单已创建，但未发送
    PENDING = 'PENDING'      # 已发送到交易所，等待确认
    WORKING = 'WORKING'      # 交易所已确认，等待成交
    PARTIALLY_FILLED = 'PARTIALLY_FILLED'  # 部分成交
    FILLED = 'FILLED'        # 完全成交
    CANCELED = 'CANCELED'    # 已取消
    REJECTED = 'REJECTED'    # 交易所拒绝
    ERROR = 'ERROR'          # 系统错误


class TradeExecutorService:
    """
    交易执行服务
    
    【订单-决策绑定机制】
    1. 接收信号时，使用decision_id作为ClientOrderID
    2. 订单执行后，将交易所订单ID与决策上下文绑定
    3. 持久化到Redis和数据库（可选）
    """
    
    def __init__(self, redis_config: Dict[str, Any], symbol: str, risk_config: Optional[Dict[str, Any]] = None):
        """
        初始化交易执行服务
        
        Args:
            redis_config: Redis配置字典
            symbol: 交易品种
            risk_config: 风控配置字典（可选）
        """
        self.redis_config = redis_config
        self.symbol = symbol
        
        # Redis客户端（文本模式，用于JSON序列化）
        self.r = redis.Redis(
            host=redis_config.get('host', 'localhost'),
            port=redis_config.get('port', 6379),
            db=redis_config.get('db', 0),
            decode_responses=True
        )
        
        # 风控管理器
        self.risk_controller = None
        if RiskController:
            try:
                self.risk_controller = RiskController(symbol, risk_config)
                logger.info("风控管理器已启用")
            except Exception as e:
                logger.warning(f"风控管理器初始化失败: {e}")
        
        # 监控服务引用（可选，用于报告风险突破）
        self.monitor_service = None
        
        # 实时订单跟踪器（Order FSM状态管理）
        # 结构: {client_order_id: {status, decision_id, signal_context, exchange_order_id, ...}}
        self.live_order_tracker: Dict[str, Dict[str, Any]] = {}
        self.tracker_lock = Lock()
        
        # 连接管理
        self.exchange_connected = False
        self.consecutive_errors = 0
        self.max_consecutive_errors = 5
        self.retry_delay_base = 1.0  # 基础重试延迟（秒）
        self.max_retry_delay = 60.0  # 最大重试延迟（秒）
        
        # Redis键名
        self.order_stream_key = f"order:{symbol}:stream"  # 实时订单流
        self.order_history_key = f"order:{symbol}:history"  # 历史订单（Sorted Set）
        self.trade_history_key = f"trade:{symbol}:history"  # 成交记录（Sorted Set）
        
        logger.info(f"交易执行服务已初始化: {symbol}")
    
    def set_monitor_service(self, monitor_service):
        """设置监控服务引用（用于报告风险突破）"""
        self.monitor_service = monitor_service
        if self.risk_controller:
            self.risk_controller.set_monitor_service(monitor_service)
    
    def send_signal(self, signal_record: Dict[str, Any]) -> Optional[str]:
        """
        接收交易信号并发送订单到交易所
        
        【完整流程】
        1. 交易信号预处理
        2. 实时风控检查（Pre-Trade Risk Check）
        3. 创建订单（NEW状态）
        4. 发送到交易所（带重试机制）
        5. 更新订单状态（PENDING -> WORKING）
        
        Args:
            signal_record: 完整的交易信号记录（包含decision_id和context）
            
        Returns:
            str: 交易所订单ID（如果成功），否则返回None
        """
        try:
            action = signal_record.get('action')
            decision_id = signal_record.get('decision_id')
            price = signal_record.get('price', 0.0)
            symbol = signal_record.get('symbol', self.symbol)
            
            if not decision_id:
                logger.error("交易信号缺少decision_id，无法绑定")
                return None
            
            # 1. 交易信号预处理（将BUY/SELL转换为交易所格式）
            if action == 'BUY':
                exchange_side = 'BUY'
                order_type = 'LIMIT' if price > 0 else 'MARKET'
            elif action == 'SELL':
                exchange_side = 'SELL'
                order_type = 'LIMIT' if price > 0 else 'MARKET'
            elif action == 'FLAT':
                # 平仓信号：取消所有工作订单并平仓
                logger.info(f"决策 {decision_id}: 收到FLAT信号，执行平仓逻辑")
                self._handle_flat_signal(signal_record)
                return None
            else:
                logger.error(f"未知的交易动作: {action}")
                return None
            
            # 2. 实时风控检查（Pre-Trade Risk Check）
            quantity = signal_record.get('quantity', 0.01)  # 默认0.01手
            atr = None
            if signal_record.get('context') and signal_record['context'].get('macro_indicators'):
                atr = signal_record['context']['macro_indicators'].get('ATR')
            
            if self.risk_controller:
                risk_ok, risk_reason = self.risk_controller.check_pre_trade_limits(
                    exchange_side, quantity, price, atr
                )
                if not risk_ok:
                    logger.warning(f"决策 {decision_id}: 风控拒绝下单。原因: {risk_reason}")
                    self._record_final_order_state(signal_record, OrderStatus.REJECTED, f"Risk Check: {risk_reason}")
                    return None
            
            # 3. 创建新订单并加入跟踪器（NEW状态）
            client_order_id = decision_id  # 使用decision_id作为ClientOrderID
            
            order_data = {
                'client_order_id': client_order_id,
                'decision_id': decision_id,
                'status': OrderStatus.NEW.value,
                'symbol': symbol,
                'side': exchange_side,
                'order_type': order_type,
                'price': price,
                'quantity': quantity,
                'signal_context': signal_record,
                'created_time': time.time(),
                'last_update_time': time.time()
            }
            
            self._update_order_state(client_order_id, order_data)
            
            # 4. 发送订单到交易所（带重试机制）
            exchange_order_id = None
            max_retries = 3
            
            for retry in range(max_retries):
                try:
                    # 检查连接状态
                    if not self.exchange_connected and self.consecutive_errors >= self.max_consecutive_errors:
                        logger.error("交易所连接异常，尝试重连...")
                        if not self._reconnect_exchange():
                            time.sleep(self._get_retry_delay(retry))
                            continue
                    
                    # 发送订单（模拟）
                    # 实际系统中：exchange_order_id = self.exchange_client.place_order(order_data)
                    exchange_order_id = f"EX-{symbol}-{int(time.time() * 1000)}"
                    
                    # 订单发送成功，状态进入PENDING
                    self._update_order_state(client_order_id, {
                        'status': OrderStatus.PENDING.value,
                        'exchange_order_id': exchange_order_id,
                        'last_update_time': time.time()
                    })
                    
                    # 重置错误计数
                    self.consecutive_errors = 0
                    self.exchange_connected = True
                    
                    logger.info(f"📤 订单发送成功: {action} @ {price:.2f} | Decision ID: {decision_id} | Exchange Order ID: {exchange_order_id}")
                    break
                    
                except Exception as e:
                    self.consecutive_errors += 1
                    logger.error(f"订单发送失败（重试 {retry + 1}/{max_retries}）: {e}")
                    
                    if retry < max_retries - 1:
                        delay = self._get_retry_delay(retry)
                        time.sleep(delay)
                    else:
                        # 所有重试失败
                        self._update_order_state(client_order_id, {
                            'status': OrderStatus.ERROR.value,
                            'error': str(e),
                            'last_update_time': time.time()
                        })
                        self._record_final_order_state(signal_record, OrderStatus.ERROR, f"API Error: {e}")
                        return None
            
            # 5. 模拟订单进入WORKING状态（实际由交易所确认）
            # 在实际系统中，这应该由handle_execution_report处理
            if exchange_order_id:
                # 模拟：订单成功进入WORKING状态
                self._update_order_state(client_order_id, {
                    'status': OrderStatus.WORKING.value,
                    'last_update_time': time.time()
                })
                
                # 记录订单到Redis Stream（实时监控）
                self._publish_order_update(client_order_id)
            
            return exchange_order_id
            
        except Exception as e:
            logger.error(f"发送交易信号失败: {e}")
            return None
    
    def _handle_flat_signal(self, signal_record: Dict[str, Any]):
        """处理平仓信号"""
        # 取消所有工作订单
        with self.tracker_lock:
            working_orders = [
                order_id for order_id, order in self.live_order_tracker.items()
                if order.get('status') in [OrderStatus.PENDING.value, OrderStatus.WORKING.value]
            ]
        
        for order_id in working_orders:
            self._cancel_order(order_id, "FLAT signal received")
        
        logger.info(f"平仓信号处理完成: 已取消 {len(working_orders)} 个工作订单")
    
    def _cancel_order(self, client_order_id: str, reason: str):
        """取消订单"""
        # 实际系统中：调用交易所API取消订单
        self._update_order_state(client_order_id, {
            'status': OrderStatus.CANCELED.value,
            'cancel_reason': reason,
            'last_update_time': time.time()
        })
    
    def _reconnect_exchange(self) -> bool:
        """重连交易所（模拟）"""
        # 实际系统中：重新初始化交易所连接
        try:
            # self.exchange_client.reconnect()
            self.exchange_connected = True
            self.consecutive_errors = 0
            logger.info("交易所重连成功")
            return True
        except Exception as e:
            logger.error(f"交易所重连失败: {e}")
            return False
    
    def _get_retry_delay(self, retry_count: int) -> float:
        """计算重试延迟（指数退避）"""
        delay = min(self.retry_delay_base * (2 ** retry_count), self.max_retry_delay)
        return delay
    
    def _update_order_state(self, client_order_id: str, updates: Dict[str, Any]):
        """原子性地更新订单在实时跟踪器中的状态"""
        with self.tracker_lock:
            if client_order_id not in self.live_order_tracker:
                self.live_order_tracker[client_order_id] = {}
            self.live_order_tracker[client_order_id].update(updates)
            self.live_order_tracker[client_order_id]['last_update_time'] = time.time()
    
    def _publish_order_update(self, client_order_id: str):
        """发布订单更新到Redis Stream"""
        try:
            order = self.live_order_tracker.get(client_order_id, {})
            order_json = json.dumps({
                'order_id': order.get('exchange_order_id', ''),
                'client_order_id': client_order_id,
                'decision_id': order.get('decision_id', ''),
                'action': order.get('side', ''),
                'price': order.get('price', 0.0),
                'status': order.get('status', ''),
                'timestamp': time.time()
            }, ensure_ascii=False)
            
            self.r.xadd(
                self.order_stream_key,
                {'order_json': order_json},
                id='*',
                maxlen=1000,
                approximate=True
            )
        except Exception as e:
            logger.error(f"发布订单更新失败: {e}")
    
    def _is_valid_transition(self, current_status: str, new_status: str) -> bool:
        """
        检查订单状态迁移是否有效
        
        【状态迁移规则】
        NEW -> PENDING, REJECTED, ERROR
        PENDING -> WORKING, REJECTED, ERROR
        WORKING -> FILLED, PARTIALLY_FILLED, CANCELED, REJECTED, ERROR
        PARTIALLY_FILLED -> FILLED, CANCELED, REJECTED, ERROR
        最终状态（FILLED, CANCELED, REJECTED, ERROR）不可再迁移
        """
        # 最终状态不可再迁移
        final_states = [OrderStatus.FILLED.value, OrderStatus.CANCELED.value, 
                       OrderStatus.REJECTED.value, OrderStatus.ERROR.value]
        if current_status in final_states:
            return False
        
        # 定义有效迁移
        valid_transitions = {
            OrderStatus.NEW.value: [OrderStatus.PENDING.value, OrderStatus.REJECTED.value, OrderStatus.ERROR.value],
            OrderStatus.PENDING.value: [OrderStatus.WORKING.value, OrderStatus.REJECTED.value, OrderStatus.ERROR.value],
            OrderStatus.WORKING.value: [OrderStatus.FILLED.value, OrderStatus.PARTIALLY_FILLED.value, 
                                       OrderStatus.CANCELED.value, OrderStatus.REJECTED.value, OrderStatus.ERROR.value],
            OrderStatus.PARTIALLY_FILLED.value: [OrderStatus.FILLED.value, OrderStatus.CANCELED.value, 
                                                 OrderStatus.REJECTED.value, OrderStatus.ERROR.value]
        }
        
        allowed_next = valid_transitions.get(current_status, [])
        return new_status in allowed_next
    
    def handle_execution_report(self, execution_report: Dict[str, Any]):
        """
        处理交易所返回的执行报告（成交、取消、拒绝等）
        
        【关键功能】
        1. 通过ClientOrderID查找原始决策上下文
        2. 创建最终的订单-决策绑定记录
        3. 持久化到Redis和数据库
        
        Args:
            execution_report: 交易所执行报告，包含：
                - client_order_id: 客户端订单ID（即decision_id）
                - exchange_order_id: 交易所订单ID
                - status: 订单状态（FILLED, CANCELED, REJECTED等）
                - execution_price: 成交价格
                - executed_quantity: 成交数量
                - timestamp: 执行时间
        """
        """
        处理来自交易所的执行报告，驱动订单状态机（Order FSM）
        
        【状态机逻辑】
        1. 验证状态迁移有效性
        2. 更新订单状态
        3. 处理最终状态（持久化、更新仓位、移除跟踪）
        4. 处理部分成交
        """
        try:
            client_order_id = execution_report.get('client_order_id')
            exchange_order_id = execution_report.get('exchange_order_id')
            new_status_str = execution_report.get('status', 'UNKNOWN')
            
            if not client_order_id:
                logger.error("执行报告缺少client_order_id，无法绑定决策")
                return
            
            # 1. 查找原始记录
            with self.tracker_lock:
                current_order = self.live_order_tracker.get(client_order_id)
            
            if not current_order:
                logger.warning(f"收到未知订单报告: {client_order_id}，可能已处理或为幽灵订单")
                return
            
            # 2. 验证状态迁移有效性
            current_status = current_order.get('status', OrderStatus.NEW.value)
            if not self._is_valid_transition(current_status, new_status_str):
                logger.error(f"订单 {client_order_id} 无效状态迁移: {current_status} -> {new_status_str}")
                return
            
            # 3. 更新订单状态
            self._update_order_state(client_order_id, {
                'status': new_status_str,
                'last_update_time': time.time()
            })
            
            # 4. 处理最终状态
            if new_status_str in [OrderStatus.FILLED.value, OrderStatus.CANCELED.value, 
                                 OrderStatus.REJECTED.value, OrderStatus.ERROR.value]:
                
                # 提取原始决策上下文
                signal_context = current_order.get('signal_context', {})
                decision_id = signal_context.get('decision_id', client_order_id)
                decision_context = signal_context.get('context', {})
                
                # 如果是成交，更新仓位和风控
                if new_status_str == OrderStatus.FILLED.value:
                    side = current_order.get('side', '')
                    quantity = current_order.get('quantity', 0.0)
                    if self.risk_controller:
                        self.risk_controller.update_position(side, quantity)
                        self.risk_controller.record_trade()
                
                # 创建最终持久化记录
                self._record_final_order_state(signal_context, new_status_str, 
                                             execution_report.get('reason', 'Executed'))
                
                # 从实时跟踪器中移除
                with self.tracker_lock:
                    if client_order_id in self.live_order_tracker:
                        del self.live_order_tracker[client_order_id]
                
                logger.info(f"订单 {client_order_id} 达到最终状态 {new_status_str}，已移除跟踪")
            
            # 5. 处理部分成交
            elif new_status_str == OrderStatus.PARTIALLY_FILLED.value:
                filled_quantity = execution_report.get('filled_quantity', 0.0)
                side = current_order.get('side', '')
                if self.risk_controller:
                    self.risk_controller.update_position(side, filled_quantity)
                
                logger.info(f"订单 {client_order_id} 部分成交: {filled_quantity}")
            
        except Exception as e:
            logger.error(f"处理执行报告失败: {e}")
            
            
        except Exception as e:
            logger.error(f"处理执行报告失败: {e}")
    
    def _record_final_order_state(self, signal_context: Dict[str, Any], final_status: str, reason: str):
        """
        创建最终持久化记录，包含决策上下文和成交结果
        
        Args:
            signal_context: 原始信号上下文
            final_status: 最终状态
            reason: 状态原因
        """
        try:
            decision_id = signal_context.get('decision_id', 'UNKNOWN')
            decision_context = signal_context.get('context', {})
            
            final_trade_record = {
                'trade_id': f"TRADE-{int(time.time() * 1000)}",
                'decision_id': decision_id,
                'final_status': final_status,
                'final_reason': reason,
                'execution_time': time.time(),
                'execution_time_ms': int(time.time() * 1000),
                'execution_price': signal_context.get('price', 0.0),
                'executed_quantity': signal_context.get('quantity', 0.0),
                'action': signal_context.get('action', 'UNKNOWN'),
                'symbol': self.symbol,
                
                # 决策上下文（完整绑定）
                'decision_context': decision_context,
                'kline_time_m1': signal_context.get('kline_time_m1', 0),
                'tick_time_ms': signal_context.get('tick_time_ms', 0),
                'reason': signal_context.get('reason', ''),
                
                # 原始信号信息
                'original_signal': {
                    'timestamp': signal_context.get('timestamp', 0),
                    'price': signal_context.get('price', 0.0),
                    'fsm_state': signal_context.get('current_state', 'UNKNOWN'),
                }
            }
            
            # 持久化到Redis
            self._persist_final_record(final_trade_record)
            
            logger.success(f"💰 最终记录: Decision {decision_id} 状态: {final_status}。已持久化。")
            
        except Exception as e:
            logger.error(f"记录最终订单状态失败: {e}")
    
    def _persist_final_record(self, final_record: Dict[str, Any]):
        """
        持久化最终的订单-决策绑定记录
        
        【存储位置】
        - Redis Sorted Set: `trade:{symbol}:history` - 历史成交记录
        - Redis Hash: `trade:{symbol}:{trade_id}` - 详细交易记录（可选）
        
        Args:
            final_record: 最终的订单-决策绑定记录
        """
        try:
            trade_id = final_record.get('trade_id')
            execution_time_ms = final_record.get('execution_time_ms', int(time.time() * 1000))
            
            # 序列化为JSON
            record_json = json.dumps(final_record, ensure_ascii=False)
            
            # 1. 保存到Sorted Set（历史查询，按时间排序）
            self.r.zadd(
                self.trade_history_key,
                {record_json: execution_time_ms}
            )
            
            # 2. 保存到Hash（按trade_id快速查询）
            trade_hash_key = f"trade:{self.symbol}:{trade_id}"
            self.r.hset(trade_hash_key, mapping={
                'trade_id': trade_id,
                'decision_id': final_record.get('decision_id', ''),
                'record_json': record_json
            })
            self.r.expire(trade_hash_key, 7 * 24 * 60 * 60)  # 7天过期
            
            # 3. 滚动删除（保留最近30天的记录）
            thirty_days_ago = execution_time_ms - (30 * 24 * 60 * 60 * 1000)
            self.r.zremrangebyscore(self.trade_history_key, '-inf', thirty_days_ago)
            
            logger.debug(f"订单-决策绑定记录已持久化: Trade ID={trade_id}, Decision ID={final_record.get('decision_id')}")
            
        except Exception as e:
            logger.error(f"持久化订单记录失败: {e}")
    
    def get_trade_by_id(self, trade_id: str) -> Optional[Dict[str, Any]]:
        """
        根据Trade ID查询订单-决策绑定记录
        
        Args:
            trade_id: 交易ID
            
        Returns:
            订单-决策绑定记录，如果不存在则返回None
        """
        try:
            trade_hash_key = f"trade:{self.symbol}:{trade_id}"
            record_json = self.r.hget(trade_hash_key, 'record_json')
            
            if record_json:
                return json.loads(record_json)
            return None
            
        except Exception as e:
            logger.error(f"查询交易记录失败: {e}")
            return None
    
    def get_trades_by_decision_id(self, decision_id: str) -> List[Dict[str, Any]]:
        """
        根据Decision ID查询所有相关订单
        
        Args:
            decision_id: 决策ID
            
        Returns:
            订单记录列表
        """
        try:
            # 从Sorted Set中查找（需要遍历，实际系统中可以使用索引）
            all_trades = self.r.zrange(self.trade_history_key, 0, -1, withscores=False)
            
            matching_trades = []
            for trade_json in all_trades:
                try:
                    trade = json.loads(trade_json)
                    if trade.get('decision_id') == decision_id:
                        matching_trades.append(trade)
                except Exception as e:
                    logger.warning(f"解析交易记录失败: {e}")
                    continue
            
            return matching_trades
            
        except Exception as e:
            logger.error(f"按决策ID查询交易失败: {e}")
            return []
    
    def get_trades_by_time_range(self, start_time_ms: int, end_time_ms: int) -> List[Dict[str, Any]]:
        """
        按时间范围查询订单记录
        
        Args:
            start_time_ms: 开始时间（毫秒）
            end_time_ms: 结束时间（毫秒）
            
        Returns:
            订单记录列表
        """
        try:
            trades_data = self.r.zrangebyscore(
                self.trade_history_key,
                start_time_ms,
                end_time_ms,
                withscores=False
            )
            
            trades = []
            for trade_json in trades_data:
                try:
                    trade = json.loads(trade_json)
                    trades.append(trade)
                except Exception as e:
                    logger.warning(f"解析交易记录失败: {e}")
                    continue
            
            return trades
            
        except Exception as e:
            logger.error(f"按时间范围查询交易失败: {e}")
            return []

