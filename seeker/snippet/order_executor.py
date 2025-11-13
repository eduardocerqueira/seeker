#date: 2025-11-13T17:02:54Z
#url: https://api.github.com/gists/4c793df7bd6fb44895f49714b67dd982
#owner: https://api.github.com/users/wangwei334455

"""
L1外部通信层 - 订单执行器

职责：
1. 监听L2发送的交易指令（通过Redis List）
2. 通过gRPC调用Windows中继服务执行订单
3. 将订单结果反馈给L2（通过Redis List）

🔴 架构说明：
- Linux后端不直接连接MT5
- 通过gRPC（端口50051）调用Windows中继服务
- Windows中继服务负责连接MT5并执行订单
"""
import time
import json
import redis
from typing import Dict, List, Optional, Any
from datetime import datetime
from threading import Thread, Event
from loguru import logger

# 🔴 安全机制：导入环境检查模块
try:
    from src.trading.utils.env_check import (
        is_production_mode, 
        require_production_mode,
        get_env_info,
        log_env_status
    )
except ImportError:
    # 如果导入失败，提供降级方案（默认允许，但记录警告）
    logger.warning("⚠️ 无法导入环境检查模块，使用降级方案（允许所有交易）")
    def is_production_mode():
        return True  # 降级：允许交易
    def require_production_mode(func_name: str = "执行交易"):
        return True
    def get_env_info():
        return {'env': 'UNKNOWN', 'is_production': True}
    def log_env_status():
        logger.warning("⚠️ 环境检查模块未加载")

# 🔴 修复：Linux后端不需要导入MT5，MT5在Windows中继服务上
# Linux后端通过gRPC连接Windows MT5中继服务，中继服务负责连接MT5
# 移除MT5导入，避免不必要的警告

# Redis配置（从config模块导入）
try:
    from config.redis_config import REDIS_CONFIG, REDIS_KEYS
except ImportError:
    REDIS_CONFIG = {
        'host': 'localhost',
        'port': 6379,
        'db': 0,
        'decode_responses': True
    }
    REDIS_KEYS = {}

# L2发送指令到L1的队列（Redis List）
L2_ORDER_QUEUE = 'l2:order:commands'
# L1反馈订单结果给L2的队列（Redis List）
L1_FEEDBACK_QUEUE = 'l1:order:feedback'
# L3人工订单指令队列（Redis Stream）
L3_MANUAL_COMMANDS_STREAM = 'l3:manual:commands'


class OrderExecutor:
    """
    L1订单执行器
    
    职责：
    1. 监听L2发送的交易指令（通过Redis List）
    2. 通过gRPC调用Windows中继服务执行订单
    3. 将订单结果反馈给L2（通过Redis List）
    
    🔴 架构说明：
    - Linux后端不直接连接MT5
    - 通过gRPC（端口50051）调用Windows中继服务
    - Windows中继服务负责连接MT5并执行订单
    """
    
    # 订单类型 (参考MT5)
    ORDER_TYPE_BUY = 0          # 买入市价单
    ORDER_TYPE_SELL = 1         # 卖出市价单
    ORDER_TYPE_BUY_LIMIT = 2    # 买入限价单
    ORDER_TYPE_SELL_LIMIT = 3   # 卖出限价单
    
    # 订单状态
    ORDER_STATE_STARTED = 0     # 已启动
    ORDER_STATE_PLACED = 1      # 已下单
    ORDER_STATE_CANCELED = 2    # 已取消
    ORDER_STATE_PARTIAL = 3     # 部分成交
    ORDER_STATE_FILLED = 4      # 完全成交
    ORDER_STATE_REJECTED = 5    # 已拒绝
    
    # 成交类型
    DEAL_TYPE_BUY = 0           # 买入
    DEAL_TYPE_SELL = 1          # 卖出
    DEAL_TYPE_BALANCE = 2       # 余额操作
    
    # 进场类型
    DEAL_ENTRY_IN = 0           # 入场
    DEAL_ENTRY_OUT = 1          # 出场
    DEAL_ENTRY_INOUT = 2        # 反转
    
    def __init__(self, 
                 symbol: str = "BTCUSDm",
                 redis_host: Optional[str] = None,
                 redis_port: Optional[int] = None,
                 redis_db: Optional[int] = None):
        """
        初始化订单执行器
        
        Args:
            symbol: 交易品种
            redis_host: Redis主机地址（默认从配置读取）
            redis_port: Redis端口（默认从配置读取）
            redis_db: Redis数据库编号（默认从配置读取）
        """
        self.symbol = symbol
        self.stop_event = Event()
        
        # 🔴 安全机制：记录环境状态
        log_env_status()
        
        # Redis连接
        redis_host = redis_host or REDIS_CONFIG.get('host', 'localhost')
        redis_port = redis_port or REDIS_CONFIG.get('port', 6379)
        redis_db = redis_db or REDIS_CONFIG.get('db', 0)
        
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            decode_responses=True
        )
        
        # 测试Redis连接
        try:
            self.redis_client.ping()
            logger.info(f"OrderExecutor: Redis连接成功 {redis_host}:{redis_port}")
        except Exception as e:
            logger.error(f"OrderExecutor: Redis连接失败: {e}")
            raise
        
        # Redis键前缀
        self.ORDERS_KEY = REDIS_KEYS.get('orders', 'trading:orders')
        self.POSITIONS_KEY = REDIS_KEYS.get('positions', 'trading:positions')
        self.DEALS_KEY = REDIS_KEYS.get('deals', 'trading:deals')
        self.TICKET_COUNTER = REDIS_KEYS.get('ticket_counter', 'trading:ticket_counter')
        
        # Windows中继服务gRPC配置（Linux后端不直接连接MT5，通过gRPC调用中继服务）
        try:
            from config.relay_config import RELAY_SERVICE, GRPC_ADDRESS
            self.grpc_host = RELAY_SERVICE['host']
            self.grpc_port = RELAY_SERVICE['grpc_port']
            self.grpc_address = GRPC_ADDRESS
            logger.info(f"OrderExecutor: Windows中继服务gRPC配置 - {self.grpc_address}")
        except ImportError:
            # 如果配置文件不存在，使用默认值
            self.grpc_host = '192.168.10.131'
            self.grpc_port = 50051
            self.grpc_address = f"{self.grpc_host}:{self.grpc_port}"
            logger.warning(f"OrderExecutor: 未找到relay_config，使用默认Windows中继服务 - {self.grpc_address}")
        
        # 初始化gRPC客户端（先初始化状态变量）
        self.grpc_client = None
        self.grpc_available = False
        
        try:
            from src.trading.services.grpc_trade_client import get_grpc_client, is_grpc_available
            if is_grpc_available():
                self.grpc_client = get_grpc_client(host=self.grpc_host, port=self.grpc_port)
                # 测试连接是否可用
                self.grpc_available = self._test_grpc_connection()
                if self.grpc_available:
                    logger.info(f"OrderExecutor: gRPC客户端初始化成功 - {self.grpc_address}")
                else:
                    logger.warning(f"OrderExecutor: gRPC客户端已创建，但连接测试失败 - {self.grpc_address}")
                    logger.warning("可能原因：Windows MT5中继服务未启动或网络不可达")
            else:
                self.grpc_client = None
                self.grpc_available = False
                logger.error("OrderExecutor: gRPC功能不可用，订单执行将失败")
                logger.error("请运行: python scripts/generate_grpc_code.py --target linux")
        except Exception as e:
            self.grpc_client = None
            self.grpc_available = False
            logger.error(f"OrderExecutor: gRPC客户端初始化失败: {e}")
            logger.error("请运行: python scripts/generate_grpc_code.py --target linux")
        
        # 🔴 修复：Linux后端不连接MT5，MT5在Windows中继服务上
        # 中继服务包含2个服务：
        # 1. gRPC服务（端口50051）- 处理查询和指令（同步），包括订单执行
        # 2. ZeroMQ服务（端口5555）- Windows端内部通信（MQL EA → Python），不用于订单执行
        # 订单执行走gRPC，ZeroMQ只用于Windows端内部事件推送
        
        # 启动监听线程（监听L2指令）
        self.listener_thread = Thread(target=self._command_listener, daemon=True, name="OrderExecutorListener")
        self.listener_thread.start()
        logger.info("OrderExecutor: L2命令监听线程已启动")
        
        # 启动人工订单监听线程（监听L3人工指令）
        self.manual_listener_thread = Thread(target=self._manual_command_listener, daemon=True, name="OrderExecutorManualListener")
        self.manual_listener_thread.start()
        logger.info("OrderExecutor: L3人工订单监听线程已启动")
    
    def _test_grpc_connection(self) -> bool:
        """
        测试Windows中继服务gRPC服务是否可用（带自动重连）
        
        🔴 架构说明（根据docs/系统架构/后端架构.md）：
        - gRPC服务（端口50051）：处理查询和指令（同步），包括订单执行
        - ZeroMQ服务（端口5555）：Windows端内部通信（MQL EA → Python），不用于订单执行
        - 订单执行走gRPC，ZeroMQ只用于Windows端内部事件推送
        """
        if self.grpc_client is None:
            return False
        try:
            # 🔴 修复：确保连接可用（自动重连）
            self.grpc_client._ensure_connected(retry_count=2)
            
            # 尝试获取持仓（轻量级测试）
            result = self.grpc_client.get_positions(account_id='', symbol='', ticket=0, magic=0)
            if result.get('success') is not None:
                logger.info(f"✓ Windows中继服务gRPC可用: {self.grpc_address}")
                return True
            else:
                logger.warning(f"Windows中继服务gRPC响应异常: {result.get('message', '未知错误')}")
                return False
        except Exception as e:
            logger.warning(f"Windows中继服务gRPC不可用: {e}")
            # 🔴 修复：连接失败时，清除连接状态以便下次重连
            if self.grpc_client:
                self.grpc_client._channel = None
                self.grpc_client._stub = None
            return False
    
    def get_relay_service_status(self) -> Dict[str, Any]:
        """
        获取Windows中继服务状态（gRPC和ZeroMQ）
        
        🔴 架构说明（根据docs/系统架构/后端架构.md）：
        - Windows中继服务 = gRPC服务（50051）+ ZeroMQ服务（5555）
        - gRPC服务：处理查询和指令（同步），包括订单执行
        - ZeroMQ服务：Windows端内部通信（MQL EA → Python），不用于订单执行
        - 订单执行走gRPC，ZeroMQ只用于Windows端内部事件推送
        
        Returns:
            包含gRPC和ZeroMQ服务状态的字典
        """
        status = {
            'grpc': {
                'available': self.grpc_available,
                'address': self.grpc_address,
                'host': self.grpc_host,
                'port': self.grpc_port,
                'description': '处理查询和指令（同步），包括订单执行',
                'used_for': '订单执行、持仓查询、账户查询'
            },
            'zmq': {
                'available': None,  # ZeroMQ是Windows端内部服务，Linux端不直接连接
                'address': f"{self.grpc_host}:5555",  # ZeroMQ默认端口
                'description': 'Windows端内部通信（MQL EA → Python），不用于订单执行',
                'used_for': 'Windows端内部事件推送，Linux端通过Redis Pub/Sub接收'
            },
            'relay_host': self.grpc_host,
            'note': 'Windows中继服务包含gRPC和ZeroMQ两个服务，订单执行走gRPC，中继服务负责连接MT5'
        }
        return status
    
    def _execute_via_grpc(self, action: str, price: float, volume: float, sl: float = 0.0, tp: float = 0.0) -> Dict[str, Any]:
        """
        通过Windows MT5 gRPC服务执行订单
        
        Args:
            action: 'BUY' 或 'SELL'
            price: 价格（市价单为0）
            volume: 交易手数
            sl: 止损价
            tp: 止盈价
            
        Returns:
            订单执行结果字典
        """
        if not self.grpc_available or self.grpc_client is None:
            return {
                'status': 'RELAY_UNAVAILABLE',
                'comment': 'gRPC服务不可用'
            }
        
        try:
            # 转换订单类型
            order_type = 0 if action == 'BUY' else 1  # 0=BUY, 1=SELL
            
            # 通过gRPC发送订单
            # 确保所有参数都是正确的数字类型（防御性编程）
            try:
                price_float = float(price) if price is not None else 0.0
            except (ValueError, TypeError):
                price_float = 0.0
            
            try:
                volume_float = float(volume) if volume is not None else 0.01
            except (ValueError, TypeError):
                volume_float = 0.01
            
            try:
                sl_float = float(sl) if sl is not None else 0.0
            except (ValueError, TypeError):
                sl_float = 0.0
            
            try:
                tp_float = float(tp) if tp is not None else 0.0
            except (ValueError, TypeError):
                tp_float = 0.0
            
            # 🔴 安全机制：检查是否为生产模式
            try:
                require_production_mode(f"执行订单 ({action} {self.symbol} {volume_float}手)")
            except EnvironmentError as e:
                logger.error(f"OrderExecutor: {str(e)}")
                return {
                    'status': 'BLOCKED',
                    'comment': f'非生产环境，订单执行已阻止。当前环境: {get_env_info().get("env", "UNKNOWN")}',
                    'error': 'ENVIRONMENT_BLOCKED'
                }
            
            logger.info(f"OrderExecutor: 通过gRPC执行订单 - {action} {self.symbol} {volume_float}手")
            
            # 🔴 修复：确保gRPC连接可用（自动重连）
            try:
                # gRPC Trade Client 的 _ensure_connected 会自动重连
            result = self.grpc_client.send_trade(
                account_id='',  # 空字符串表示使用默认账户
                symbol=self.symbol,
                order_type=order_type,
                volume=volume_float,
                price=price_float if price_float > 0 else 0.0,  # 市价单为0
                stop_loss=sl_float,
                take_profit=tp_float,
                magic=202409,  # 默认魔术号
                comment=f'OrderExecutor_{action}_{self.symbol}'
            )
            except Exception as grpc_error:
                # 🔴 修复：gRPC调用失败，尝试重连并重试一次
                logger.warning(f"OrderExecutor: gRPC调用失败，尝试重连: {grpc_error}")
                try:
                    # 强制重连（通过重新初始化客户端）
                    self.grpc_client._channel = None
                    self.grpc_client._stub = None
                    self.grpc_client._ensure_connected(retry_count=3)
                    
                    # 重试一次
                    result = self.grpc_client.send_trade(
                        account_id='',
                        symbol=self.symbol,
                        order_type=order_type,
                        volume=volume_float,
                        price=price_float if price_float > 0 else 0.0,
                        stop_loss=sl_float,
                        take_profit=tp_float,
                        magic=202409,
                        comment=f'OrderExecutor_{action}_{self.symbol}'
                    )
                    logger.info("OrderExecutor: gRPC重连后订单执行成功")
                except Exception as retry_error:
                    logger.error(f"OrderExecutor: gRPC重连后仍然失败: {retry_error}")
                    self.grpc_available = False
                    return {
                        'status': 'RELAY_UNAVAILABLE',
                        'comment': f'gRPC服务不可用: {retry_error}'
                    }
            
            if result.get('success'):
                # 订单执行成功
                return {
                    'status': 'SUCCESS',
                    'order_id': result.get('order_ticket', 0),
                    'deal_id': result.get('position_ticket', 0),
                    'volume': result.get('volume', volume),
                    'price': result.get('price', 0.0),
                    'bid': result.get('price', 0.0),
                    'ask': result.get('price', 0.0),
                    'comment': result.get('message', ''),
                    'retcode': result.get('retcode', 0)
                }
            else:
                # 订单被拒绝
                return {
                    'status': 'REJECTED',
                    'retcode': result.get('retcode', -1),
                    'comment': result.get('message', '订单被拒绝'),
                    'error': result.get('message', '未知错误')
            }
        except Exception as e:
            # gRPC错误
            logger.error(f"OrderExecutor: gRPC执行订单异常: {e}")
            return {
                'status': 'RELAY_UNAVAILABLE',
                'comment': f'gRPC错误: {str(e)}'
            }
    
    def _command_listener(self):
        """
        后台线程：阻塞监听Redis List中的L2交易指令
        
        使用BRPOP实现低延迟阻塞读取（10ms超时）
        """
        logger.info("OrderExecutor: 命令监听线程已启动")
        
        while not self.stop_event.is_set():
            try:
                # 低延迟BRPOP：阻塞读取L2发送的命令，BLOCK=0.01秒（10ms）确保低延迟
                # 使用List而非Stream，因为List的LPOP/RPUSH在作为队列时延迟最低
                response = self.redis_client.brpop(L2_ORDER_QUEUE, timeout=0.01)
                
                if response:
                    _, command_json = response
                    command = json.loads(command_json)
                    
                    # 关键执行：处理指令
                    self._execute_command(command)
                
            except Exception as e:
                logger.error(f"OrderExecutor监听器错误: {e}")
                time.sleep(0.1)
        
        logger.info("OrderExecutor: 命令监听线程已停止")
    
    def _manual_command_listener(self):
        """
        后台线程：监听L3发送的人工订单指令（通过Redis Stream）
        
        使用XREAD实现低延迟阻塞读取
        启动时只处理新消息，跳过历史消息
        """
        logger.info("OrderExecutor: 人工订单监听线程已启动")
        
        # 使用文本模式的Redis客户端
        r_text = redis.Redis(
            host=REDIS_CONFIG.get('host', 'localhost'),
            port=REDIS_CONFIG.get('port', 6379),
            db=REDIS_CONFIG.get('db', 0),
            decode_responses=True
        )
        
        # 🚀 重要：启动时只处理新消息，跳过历史消息
        # 使用 '$' 表示只读取启动后新产生的消息，不处理历史消息
        # 这样可以避免启动时重复执行历史订单
        last_id = '$'  # '$' 表示只读取新消息，不读取历史消息
        logger.info("OrderExecutor: 人工订单监听器已启动，将只处理启动后的新订单（跳过历史订单）")
        
        while not self.stop_event.is_set():
            try:
                # 从Redis Stream读取人工订单指令（阻塞读取，超时100ms）
                messages = r_text.xread({L3_MANUAL_COMMANDS_STREAM: last_id}, count=1, block=100)
                
                if messages:
                    for stream, msgs in messages:
                        for msg_id, msg_data in msgs:
                            # 解析指令
                            command = dict(msg_data)
                            
                            logger.info(f"OrderExecutor: 收到人工订单指令 - {command.get('action')}, volume={command.get('volume')}")
                            
                            # 执行指令（使用与L2指令相同的处理逻辑）
                            self._execute_command(command)
                            
                            # 更新最后处理的ID
                            last_id = msg_id
                
            except Exception as e:
                logger.error(f"OrderExecutor: 人工订单监听错误: {e}")
                time.sleep(0.1)
        
        logger.info("OrderExecutor: 人工订单监听线程已停止")
    
    def _execute_command(self, command: Dict[str, Any]):
        """
        解析指令并通过gRPC调用Windows中继服务执行订单
        
        🔴 架构说明：
        - 不再直接调用MT5 API
        - 通过gRPC调用Windows中继服务
        - Windows中继服务负责连接MT5并执行订单
        
        Args:
            command: L2发送的交易指令字典
        """
        action = command.get('action')
        # Redis Stream 返回的数据都是字符串，需要转换为数字类型
        # 使用更安全的转换方式，处理空字符串、None等情况
        try:
            price_val = command.get('price', 0.0)
            price = float(price_val) if price_val and str(price_val).strip() else 0.0
        except (ValueError, TypeError):
            price = 0.0
        
        try:
            volume_val = command.get('volume', 0.01)
            volume = float(volume_val) if volume_val and str(volume_val).strip() else 0.01
        except (ValueError, TypeError):
            volume = 0.01
        
        try:
            sl_val = command.get('sl', 0.0)
            sl = float(sl_val) if sl_val and str(sl_val).strip() else 0.0
        except (ValueError, TypeError):
            sl = 0.0
        
        try:
            tp_val = command.get('tp', 0.0)
            tp = float(tp_val) if tp_val and str(tp_val).strip() else 0.0
        except (ValueError, TypeError):
            tp = 0.0
        
        logger.info(f"OrderExecutor: 收到指令 - action={action}, price={price}, volume={volume}")
        
        try:
            if action == 'BUY':
                result = self._execute_buy(price, volume, sl, tp)
                self._send_feedback(action, result)
            elif action == 'SELL':
                result = self._execute_sell(price, volume, sl, tp)
                self._send_feedback(action, result)
            elif action == 'CLOSE_ALL':
                result = self._close_all_positions()
                self._send_feedback(action, result)
            else:
                logger.warning(f"OrderExecutor: 未知指令类型: {action}")
                self._send_feedback(action, {
                    'status': 'FAILED',
                    'comment': f'Unknown action: {action}'
                })
        except Exception as e:
            logger.error(f"OrderExecutor: 执行指令时发生错误: {e}")
            self._send_feedback(action, {
                'status': 'FATAL_ERROR',
                'comment': str(e)
            })
    
    def _execute_buy(self, price: float, volume: float, sl: float = 0.0, tp: float = 0.0) -> Dict[str, Any]:
        """
        执行买入订单
        
        统一使用Windows MT5中继服务（MT5在Windows上，不在Linux上）
        """
        # 统一使用Windows MT5 gRPC服务
        if not self.grpc_available:
            # 尝试重新测试连接
            self.grpc_available = self._test_grpc_connection()
            if not self.grpc_available:
                logger.error("OrderExecutor: Windows MT5 gRPC服务不可用，无法执行订单")
                return {
                    'status': 'RELAY_UNAVAILABLE',
                    'comment': 'Windows MT5 gRPC服务不可用'
                }
        
        # 通过gRPC执行订单
        result = self._execute_via_grpc('BUY', price, volume, sl, tp)
        
        # 如果gRPC服务失败，尝试重新测试连接
        if result and result.get('status') == 'RELAY_UNAVAILABLE':
            self.grpc_available = self._test_grpc_connection()
        
        return result
    
    def _execute_sell(self, price: float, volume: float, sl: float = 0.0, tp: float = 0.0) -> Dict[str, Any]:
        """
        执行卖出订单
        
        统一使用Windows MT5中继服务（MT5在Windows上，不在Linux上）
        """
        # 统一使用Windows MT5 gRPC服务
        if not self.grpc_available:
            # 尝试重新测试连接
            self.grpc_available = self._test_grpc_connection()
            if not self.grpc_available:
                logger.error("OrderExecutor: Windows MT5 gRPC服务不可用，无法执行订单")
                return {
                    'status': 'RELAY_UNAVAILABLE',
                    'comment': 'Windows MT5 gRPC服务不可用'
                }
        
        # 通过gRPC执行订单
        result = self._execute_via_grpc('SELL', price, volume, sl, tp)
        
        # 如果gRPC服务失败，尝试重新测试连接
        if result and result.get('status') == 'RELAY_UNAVAILABLE':
            self.grpc_available = self._test_grpc_connection()
        
        return result
    
    def _build_trade_request(self, order_type: int, price: float, volume: float, sl: float = 0.0, tp: float = 0.0) -> Dict[str, Any]:
        """
        构建交易请求结构（通过gRPC发送到Windows中继服务）
        
        🔴 修复：不再使用本地MT5，所有请求通过gRPC发送到Windows中继服务
        
        Args:
            order_type: 订单类型（0=BUY, 1=SELL）
            price: 价格
            volume: 交易量
            sl: 止损价
            tp: 止盈价
            
        Returns:
            交易请求字典（gRPC格式）
        """
        # gRPC客户端会处理请求格式转换，这里只需要基本参数
        return {
            "action": 1,  # TRADE_ACTION_DEAL (市价单)
            "symbol": self.symbol,
            "volume": volume,
            "type": order_type,  # 0=BUY, 1=SELL
            "price": price,
            "deviation": 20,  # 允许价格滑点
            "magic": 202409,  # 魔术数字
            "comment": "HFT_L2_SIGNAL",
            "type_time": 0,  # ORDER_TIME_GTC
            "type_filling": 2,  # ORDER_FILLING_FOK
            "sl": sl,  # 止损
            "tp": tp,  # 止盈
        }
    
    def _simulate_order(self, action: str, price: float, volume: float, sl: float = 0.0, tp: float = 0.0) -> Dict[str, Any]:
        """
        模拟订单执行（当MT5不可用时）
        
        Args:
            action: 订单动作（'BUY' 或 'SELL'）
            price: 价格
            volume: 交易量
            sl: 止损价
            tp: 止盈价
            
        Returns:
            模拟执行结果
        """
        ticket = self._generate_ticket()
        time_msc = int(time.time() * 1000)
        
        # 创建模拟订单
        order = {
            'ticket': ticket,
            'symbol': self.symbol,
            'type': 0 if action == 'BUY' else 1,
            'volume_initial': volume,
            'volume_current': volume,
            'price_open': price,
            'price_current': price,
            'sl': sl,
            'tp': tp,
            'time_setup': time_msc // 1000,
            'time_setup_msc': time_msc,
            'time_done': time_msc // 1000,
            'time_done_msc': time_msc,
            'state': self.ORDER_STATE_FILLED,
            'magic': 202409,
            'comment': f'HFT_L2_SIGNAL_{action}',
        }
        
        # 保存订单到Redis
        self.redis_client.hset(self.ORDERS_KEY, ticket, json.dumps(order))
        
        # 更新持仓
        self._update_position(order, price, time_msc // 1000, time_msc)
        
        logger.info(f"OrderExecutor: 模拟执行订单 - {action}, ticket={ticket}, price={price}, volume={volume}")
        
        return {
            'status': 'SUCCESS',
            'order_id': ticket,
            'deal_id': ticket,
            'volume': volume,
            'price': price,
            'comment': f'Simulated {action} order'
        }
    
    def _generate_ticket(self) -> int:
        """生成唯一的订单号"""
        ticket = self.redis_client.incr(self.TICKET_COUNTER)
        if ticket == 1:
            # 首次使用，设置一个较大的初始值
            ticket = int(time.time() * 1000) % 1000000000
            self.redis_client.set(self.TICKET_COUNTER, ticket)
        return ticket
    
    def execute_order(self,
                     symbol: str,
                     order_type: int,
                     volume: float,
                     price: float,
                     time_msc: int,
                     magic: int = 0,
                     comment: str = "",
                     sl: float = 0.0,
                     tp: float = 0.0) -> Dict:
        """
        执行订单（L2策略信号 → L1订单执行）
        
        Args:
            symbol: 交易品种
            order_type: 订单类型 (0=买入, 1=卖出)
            volume: 交易手数
            price: 成交价格
            time_msc: 时间戳（毫秒）
            magic: 魔术号
            comment: 订单备注
            sl: 止损价
            tp: 止盈价
            
        Returns:
            订单字典
        """
        ticket = self._generate_ticket()
        kline_time = time_msc // 1000
        
        # 创建订单（参考MT5订单结构）
        order = {
            # 基本信息
            'ticket': ticket,
            'symbol': symbol,
            'type': order_type,
            'volume_initial': volume,
            'volume_current': 0.0,
            
            # 价格信息
            'price_open': price,
            'price_current': price,
            'price_stoplimit': 0.0,
            'sl': sl,
            'tp': tp,
            
            # 时间信息
            'time_setup': kline_time,
            'time_setup_msc': time_msc,
            'time_done': kline_time,
            'time_done_msc': time_msc,
            'time_expiration': 0,
            'kline_time': kline_time,
            
            # 状态信息
            'state': self.ORDER_STATE_FILLED,  # 市价单立即成交
            'type_filling': 0,
            'type_time': 0,
            'reason': 0,
            
            # 其他信息
            'magic': magic,
            'position_id': ticket,
            'position_by_id': 0,
            'comment': comment,
            'external_id': f'auto_{ticket}',
        }
        
        # 保存订单到Redis
        self.redis_client.hset(self.ORDERS_KEY, ticket, json.dumps(order))
        
        # 创建成交记录
        deal = self._create_deal(order, price, kline_time, time_msc)
        
        # 更新持仓
        self._update_position(order, price, kline_time, time_msc)
        
    def _send_feedback(self, action: str, result: Dict[str, Any]):
        """
        将订单执行结果推送到反馈队列（发送给L2）
        
        Args:
            action: 订单动作
            result: 执行结果字典
        """
        feedback_data = {
            'timestamp': time.time(),
            'action': action,
            'status': result.get('status', 'FAILED'),
            'order_id': result.get('order_id'),
            'deal_id': result.get('deal_id'),
            'volume': result.get('volume', 0.0),
            'price': result.get('price', 0.0),
            'comment': result.get('comment', ''),
            'retcode': result.get('retcode'),
        }
        
        # 使用Redis RPUSH（List）推送反馈
        try:
            self.redis_client.rpush(L1_FEEDBACK_QUEUE, json.dumps(feedback_data))
            logger.debug(f"OrderExecutor: 已发送反馈 - action={action}, status={feedback_data['status']}")
        except Exception as e:
            logger.error(f"OrderExecutor: 发送反馈失败: {e}")
    
    def _create_deal(self, order: Dict, price: float, kline_time: int, time_msc: int) -> Dict:
        """创建成交记录"""
        deal_ticket = self._generate_ticket()
        deal_type = self.DEAL_TYPE_BUY if order['type'] == self.ORDER_TYPE_BUY else self.DEAL_TYPE_SELL
        
        deal = {
            'ticket': deal_ticket,
            'order': order['ticket'],
            'symbol': order['symbol'],
            'type': deal_type,
            'volume': order['volume_initial'],
            'price': price,
            'entry': self.DEAL_ENTRY_IN,
            'time': kline_time,
            'time_msc': time_msc,
            'kline_time': kline_time,
            'commission': 0.0,
            'swap': 0.0,
            'profit': 0.0,
            'fee': 0.0,
            'magic': order['magic'],
            'position_id': order['position_id'],
            'reason': 0,
            'comment': order['comment'],
            'external_id': order['external_id'],
        }
        
        self.redis_client.hset(self.DEALS_KEY, deal_ticket, json.dumps(deal))
        return deal
    
    def _update_position(self, order: Dict, price: float, kline_time: int, time_msc: int):
        """更新持仓信息"""
        position_id = order['position_id']
        
        existing_position = self.redis_client.hget(self.POSITIONS_KEY, position_id)
        
        if not existing_position:
            position = {
                'ticket': position_id,
                'symbol': order['symbol'],
                'type': order['type'],
                'volume': order['volume_initial'],
                'price_open': price,
                'price_current': price,
                'sl': order['sl'],
                'tp': order['tp'],
                'time': kline_time,
                'time_msc': time_msc,
                'time_update': kline_time,
                'time_update_msc': time_msc,
                'kline_time': kline_time,
                'profit': 0.0,
                'swap': 0.0,
                'commission': 0.0,
                'magic': order['magic'],
                'comment': order['comment'],
                'external_id': order['external_id'],
                'identifier': position_id,
                'reason': 0,
            }
            self.redis_client.hset(self.POSITIONS_KEY, position_id, json.dumps(position))
    
    def _close_all_positions(self) -> Dict[str, Any]:
        """
        平仓所有头寸（简化实现）
        
        Returns:
            平仓结果字典
        """
        positions = self.get_all_positions()
        
        if not positions:
            return {
                'status': 'SUCCESS',
                'comment': 'NO_POSITIONS',
                'closed_count': 0
            }
        
        closed_count = 0
        for pos in positions:
            position_id = pos.get('ticket') or pos.get('position_id')
            if not position_id:
                continue
            
            # 获取当前价格
            # 🔴 修复：不再使用本地MT5，价格从持仓数据中获取
            # 如果持仓数据中没有price_current，使用price_open
            close_price = pos.get('price_current', pos.get('price_open', 0.0))
            
            # 计算盈亏
            if pos['type'] == self.ORDER_TYPE_BUY:
                profit = (close_price - pos['price_open']) * pos['volume']
            else:
                profit = (pos['price_open'] - close_price) * pos['volume']
            
            # 更新原订单
            original_order_data = self.redis_client.hget(self.ORDERS_KEY, position_id)
            if original_order_data:
                original_order = json.loads(original_order_data)
                original_order['close_price'] = close_price
                original_order['close_time'] = int(time.time())
                original_order['close_time_msc'] = int(time.time() * 1000)
                original_order['profit'] = profit
                original_order['state'] = self.ORDER_STATE_FILLED
                self.redis_client.hset(self.ORDERS_KEY, position_id, json.dumps(original_order))
            
            # 删除持仓
            self.redis_client.hdel(self.POSITIONS_KEY, position_id)
            closed_count += 1
            
            logger.info(f"OrderExecutor: 平仓成功 - ID={position_id}, 盈亏={profit:.2f}")
        
        return {
            'status': 'SUCCESS',
            'comment': 'CLOSED_ALL',
            'closed_count': closed_count
        }
    
    def close_position(self, position_id: int, close_price: float, kline_time: int, time_msc: int) -> Optional[Dict]:
        """平仓指定持仓"""
        position_data = self.redis_client.hget(self.POSITIONS_KEY, position_id)
        if not position_data:
            logger.error(f"持仓不存在: ID={position_id}")
            return None
        
        position = json.loads(position_data)
        
        # 计算盈亏
        if position['type'] == self.ORDER_TYPE_BUY:
            profit = (close_price - position['price_open']) * position['volume']
        else:
            profit = (position['price_open'] - close_price) * position['volume']
        
        # 更新原订单
        original_order_data = self.redis_client.hget(self.ORDERS_KEY, position_id)
        if original_order_data:
            original_order = json.loads(original_order_data)
            original_order['close_price'] = close_price
            original_order['close_time'] = kline_time
            original_order['close_time_msc'] = time_msc
            original_order['profit'] = profit
            original_order['state'] = self.ORDER_STATE_FILLED
            self.redis_client.hset(self.ORDERS_KEY, position_id, json.dumps(original_order))
            
            # 删除持仓
            self.redis_client.hdel(self.POSITIONS_KEY, position_id)
            
            # 发送平仓反馈到L2
            feedback = {
                'timestamp': time.time(),
                'action': 'CLOSE',
                'status': 'SUCCESS',
                'order_id': position_id,
                'close_price': close_price,
                'profit': profit,
                'close_time': kline_time,
                'close_time_msc': time_msc,
            }
            self.redis_client.rpush(L1_FEEDBACK_QUEUE, json.dumps(feedback))
            
            logger.info(f"平仓成功: ID={position_id}, 盈亏={profit:.2f}")
            return original_order
        
        return None
    
    def get_all_positions(self) -> List[Dict]:
        """获取所有持仓"""
        positions = []
        for position_id in self.redis_client.hkeys(self.POSITIONS_KEY):
            position_data = self.redis_client.hget(self.POSITIONS_KEY, position_id)
            if position_data:
                positions.append(json.loads(position_data))
        return positions
    
    def get_all_orders(self) -> List[Dict]:
        """获取所有订单"""
        orders = []
        for ticket in self.redis_client.hkeys(self.ORDERS_KEY):
            order_data = self.redis_client.hget(self.ORDERS_KEY, ticket)
            if order_data:
                orders.append(json.loads(order_data))
        return orders
    
    def stop(self):
        """停止订单执行器"""
        self.stop_event.set()
        # 🔴 修复：Linux后端不连接MT5，不需要关闭MT5连接
        # MT5连接由Windows中继服务管理
        logger.info("OrderExecutor: 已停止")

