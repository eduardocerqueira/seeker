#date: 2025-11-13T17:02:54Z
#url: https://api.github.com/gists/4c793df7bd6fb44895f49714b67dd982
#owner: https://api.github.com/users/wangwei334455

"""
gRPC Trade Client - Linux 端 gRPC 客户端
连接到 Windows 主机 (192.168.10.131:50051) 的 gRPC TradeService
"""
import time
import grpc
from typing import Optional, Dict, List
from pathlib import Path
import sys
from loguru import logger

# 添加 generated 目录到路径
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
GENERATED_DIR = BASE_DIR / 'generated'
if GENERATED_DIR.exists():
    # 将项目根目录和 generated 目录都添加到路径
    sys.path.insert(0, str(BASE_DIR))
    sys.path.insert(0, str(GENERATED_DIR))

# gRPC 代码导入（可选，如果未生成则使用模拟模式）
_grpc_available = False
trade_service_pb2 = None
trade_service_pb2_grpc = None

try:
    # 尝试从 generated 目录导入
    from generated import trade_service_pb2
    from generated import trade_service_pb2_grpc
    # 验证导入是否成功（检查是否为 None）
    if trade_service_pb2 is not None and trade_service_pb2_grpc is not None:
        _grpc_available = True
        logger.debug("gRPC 代码导入成功")
    else:
        logger.warning("gRPC 代码导入失败：模块为 None")
        trade_service_pb2 = None
        trade_service_pb2_grpc = None
except ImportError as e1:
    try:
        # 降级：尝试直接导入（如果 generated 在路径中）
        import trade_service_pb2
        import trade_service_pb2_grpc
        # 验证导入是否成功
        if trade_service_pb2 is not None and trade_service_pb2_grpc is not None:
            _grpc_available = True
            logger.debug("gRPC 代码导入成功（直接导入）")
        else:
            logger.warning("gRPC 代码导入失败：模块为 None（直接导入）")
            trade_service_pb2 = None
            trade_service_pb2_grpc = None
    except ImportError as e2:
        # 只在开发环境显示详细错误，生产环境静默处理
        logger.warning(f"gRPC 代码未生成或导入失败，gRPC 功能将不可用")
        logger.debug(f"第一次导入错误: {e1}")
        logger.debug(f"第二次导入错误: {e2}")
        logger.debug(f"Generated 目录: {GENERATED_DIR}, 存在: {GENERATED_DIR.exists()}")
        logger.debug("如需使用 gRPC 功能，请运行: python scripts/generate_grpc_code.py --target linux")
        # 确保设置为 None
        trade_service_pb2 = None
        trade_service_pb2_grpc = None
except Exception as e:
    # 捕获其他异常（如 ModuleNotFoundError: No module named 'grpc'）
    logger.warning(f"gRPC 代码导入异常，gRPC 功能将不可用: {e}")
    logger.debug(f"Generated 目录: {GENERATED_DIR}, 存在: {GENERATED_DIR.exists()}")
    logger.debug("如需使用 gRPC 功能，请确保已安装 grpcio: pip install grpcio grpcio-tools")
    trade_service_pb2 = None
    trade_service_pb2_grpc = None


class GrpcTradeClient:
    """
    gRPC 交易客户端
    
    用于 Linux API Gateway 连接到 Windows MT5 Relay Service 的 gRPC 服务
    """
    
    # Windows 主机地址和端口
    DEFAULT_HOST = '192.168.10.131'
    DEFAULT_PORT = 50051
    DEFAULT_TIMEOUT = 10  # 秒
    
    def __init__(self, host: str = None, port: int = None, timeout: int = None):
        """
        初始化 gRPC 客户端
        
        Args:
            host: Windows 主机地址，默认 '192.168.10.131'
            port: gRPC 端口，默认 50051
            timeout: 请求超时时间（秒），默认 10
        """
        if not _grpc_available:
            raise RuntimeError(
                "gRPC 功能不可用：gRPC 代码未生成。\n"
                "请运行: python scripts/generate_grpc_code.py --target linux"
            )
        
        self.host = host or self.DEFAULT_HOST
        self.port = port or self.DEFAULT_PORT
        self.timeout = timeout or self.DEFAULT_TIMEOUT
        self.address = f'{self.host}:{self.port}'
        
        # 连接通道和存根（延迟初始化）
        self._channel: Optional[grpc.Channel] = None
        self._stub: Optional[trade_service_pb2_grpc.TradeServiceStub] = None
        
        logger.info(f"gRPC 客户端初始化: {self.address}")
    
    def _ensure_connected(self, retry_count: int = 3):
        """
        确保 gRPC 连接已建立（带自动重连）
        
        Args:
            retry_count: 重试次数
        """
        # 首先检查 gRPC 模块是否可用
        if trade_service_pb2_grpc is None:
            raise RuntimeError(
                "gRPC 功能不可用：gRPC 代码未成功导入。\n"
                "请运行: python scripts/generate_grpc_code.py --target linux\n"
                "并确保已安装: pip install grpcio grpcio-tools"
            )
        
        # 🔴 修复：检查连接状态，如果断开则重连
        if self._channel is not None:
            try:
                state = self._channel.get_state(try_to_connect=False)
                if state == grpc.ChannelConnectivity.READY:
                    return  # 连接正常，无需重连
                elif state in (grpc.ChannelConnectivity.TRANSIENT_FAILURE, grpc.ChannelConnectivity.SHUTDOWN):
                    # 连接失败或已关闭，需要重连
                    logger.warning(f"gRPC 连接状态异常: {state}，尝试重连...")
                    self._channel = None
                    self._stub = None
            except AttributeError:
                # 如果get_state不存在，检查通道是否存在
                pass
            except Exception as e:
                logger.debug(f"检查gRPC连接状态失败: {e}")
                # 如果检查失败，尝试重新连接
                self._channel = None
                self._stub = None
        
        # 🔴 修复：自动重连机制（指数退避）
        if self._channel is None or self._stub is None:
            for attempt in range(retry_count):
            try:
                self._channel = grpc.insecure_channel(self.address)
                self._stub = trade_service_pb2_grpc.TradeServiceStub(self._channel)
                    logger.info(f"✅ 已连接到 gRPC 服务: {self.address}" + (f" (重试 {attempt+1}/{retry_count})" if attempt > 0 else ""))
                    return
            except Exception as e:
                    if attempt < retry_count - 1:
                        wait_time = 2 ** attempt  # 指数退避：1秒, 2秒, 4秒
                        logger.warning(f"gRPC 连接失败，{wait_time}秒后重试 ({attempt+1}/{retry_count}): {e}")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"连接 gRPC 服务失败（已重试{retry_count}次）: {e}")
                raise
    
    def _check_channel_state(self) -> bool:
        """
        检查通道状态（轻量级，非阻塞）
        
        Returns:
            bool: 通道是否就绪
        """
        if self._channel is None:
            # 如果通道未初始化，尝试初始化
            try:
                self._ensure_connected()
            except Exception:
                return False
        
        try:
            # 尝试获取通道状态（兼容不同gRPC版本）
            # 新版本使用 get_state(try_to_connect=False) 避免阻塞
            try:
                state = self._channel.get_state(try_to_connect=False)
                # READY 表示连接正常
                return state == grpc.ChannelConnectivity.READY
            except AttributeError:
                # 如果 get_state 不存在，检查通道是否存在
                return self._channel is not None
        except Exception as e:
            logger.debug(f"检查通道状态失败: {e}")
            return False
    
    def send_trade(
        self,
        account_id: str,
        symbol: str,
        order_type: int,  # 0=BUY, 1=SELL
        volume: float,
        price: float = 0.0,  # 0 表示市价单
        stop_loss: float = 0.0,
        take_profit: float = 0.0,
        magic: int = 202409,
        comment: str = '',
        action: int = None,  # 如果不指定，根据 price 自动判断
    ) -> Dict:
        """
        发送交易订单
        
        Args:
            account_id: 账户ID
            symbol: 交易品种 (e.g., 'BTCUSDm')
            order_type: 订单类型 (0=买入, 1=卖出)
            volume: 交易手数
            price: 委托价格 (0=市价单, >0=限价单)
            stop_loss: 止损价格
            take_profit: 止盈价格
            magic: 魔术号
            comment: 订单备注
            action: 交易操作类型 (None=自动判断, 1=市价单, 2=挂单)
            
        Returns:
            包含订单结果的字典:
            {
                'success': bool,
                'retcode': int,
                'message': str,
                'order_ticket': int,
                'position_ticket': int,
                'price': float,
                'volume': float,
            }
        """
        self._ensure_connected()
        
        # 自动判断 action
        if action is None:
            if price > 0:
                action = trade_service_pb2.TRADE_ACTION_PENDING  # 挂单
            else:
                action = trade_service_pb2.TRADE_ACTION_DEAL  # 市价单
        
        # 构建请求
        request = trade_service_pb2.TradeRequest(
            account_id=account_id,
            action=action,
            symbol=symbol,
            volume=volume,
            type=order_type,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            magic=magic,
            comment=comment or f'gRPC_{"BUY" if order_type == 0 else "SELL"}',
            timestamp=int(time.time() * 1000),
        )
        
        # 🔴 修复：确保连接可用（自动重连）
        self._ensure_connected(retry_count=3)
        
        try:
            logger.info(f"发送交易请求: {symbol} {order_type} {volume}手 @{price if price > 0 else '市价'}")
            response = self._stub.SendTrade(request, timeout=self.timeout)
            
            result = {
                'success': response.success,
                'retcode': response.retcode,
                'message': response.message,
                'order_ticket': response.order_ticket,
                'position_ticket': response.position_ticket,
                'price': response.price,
                'volume': response.volume,
            }
            
            if response.success:
                logger.info(f"订单成功: 订单号={response.order_ticket}, 持仓号={response.position_ticket}")
            else:
                logger.warning(f"订单失败: {response.message} (retcode={response.retcode})")
            
            return result
            
        except grpc.RpcError as e:
            error_msg = f"gRPC 错误: {e.code()} - {e.details()}"
            logger.error(error_msg)
            
            # 🔴 修复：如果是连接错误，清除连接状态以便重连
            if e.code() in (grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED, grpc.StatusCode.RESOURCE_EXHAUSTED):
                logger.warning("gRPC连接错误，清除连接状态以便下次重连")
                self._channel = None
                self._stub = None
            
            return {
                'success': False,
                'retcode': -1,
                'message': error_msg,
                'order_ticket': 0,
                'position_ticket': 0,
                'price': 0.0,
                'volume': 0.0,
            }
    
    def close_all_positions(
        self,
        account_id: str,
        symbol: str = '',  # 空字符串表示所有品种
        magic: int = 0,  # 0 表示所有魔术号
    ) -> Dict:
        """
        批量平仓
        
        Args:
            account_id: 账户ID
            symbol: 交易品种 (空字符串表示所有品种)
            magic: 魔术号 (0 表示所有魔术号)
            
        Returns:
            包含平仓结果的字典:
            {
                'success': bool,
                'closed_count': int,
                'message': str,
                'closed_tickets': List[int],
            }
        """
        # 🔴 修复：确保连接可用（自动重连）
        self._ensure_connected(retry_count=3)
        
        request = trade_service_pb2.CloseAllRequest(
            account_id=account_id,
            symbol=symbol,
            magic=magic,
        )
        
        try:
            logger.info(f"批量平仓请求: symbol={symbol or '全部'}, magic={magic or '全部'}")
            response = self._stub.CloseAllPositions(request, timeout=self.timeout)
            
            result = {
                'success': response.success,
                'closed_count': response.closed_count,
                'message': response.message,
                'closed_tickets': list(response.closed_tickets),
            }
            
            logger.info(f"平仓结果: 成功={response.success}, 数量={response.closed_count}")
            return result
            
        except grpc.RpcError as e:
            error_msg = f"gRPC 错误: {e.code()} - {e.details()}"
            logger.error(error_msg)
            return {
                'success': False,
                'closed_count': 0,
                'message': error_msg,
                'closed_tickets': [],
            }
    
    def modify_position_sltp(
        self,
        account_id: str,
        position_id: int,
        symbol: str,
        stop_loss: float = 0.0,
        take_profit: float = 0.0,
        magic: int = 202409,
    ) -> Dict:
        """
        修改持仓的止损/止盈
        
        Args:
            account_id: 账户ID
            position_id: 持仓ID
            symbol: 交易品种
            stop_loss: 止损价格
            take_profit: 止盈价格
            magic: 魔术号
            
        Returns:
            包含修改结果的字典
        """
        self._ensure_connected()
        
        request = trade_service_pb2.TradeRequest(
            account_id=account_id,
            action=trade_service_pb2.TRADE_ACTION_SLTP,
            symbol=symbol,
            position_id=position_id,
            stop_loss=stop_loss,
            take_profit=take_profit,
            magic=magic,
            timestamp=int(time.time() * 1000),
        )
        
        try:
            logger.info(f"修改止损/止盈: 持仓={position_id}, SL={stop_loss}, TP={take_profit}")
            response = self._stub.ModifyPositionSLTP(request, timeout=self.timeout)
            
            result = {
                'success': response.success,
                'retcode': response.retcode,
                'message': response.message,
            }
            
            if response.success:
                logger.info(f"修改成功: {response.message}")
            else:
                logger.warning(f"修改失败: {response.message}")
            
            return result
            
        except grpc.RpcError as e:
            error_msg = f"gRPC 错误: {e.code()} - {e.details()}"
            logger.error(error_msg)
            return {
                'success': False,
                'retcode': -1,
                'message': error_msg,
            }
    
    def delete_order(
        self,
        account_id: str,
        order_id: int,
        symbol: str,
        magic: int = 202409,
    ) -> Dict:
        """
        删除挂单
        
        Args:
            account_id: 账户ID
            order_id: 订单ID
            symbol: 交易品种
            magic: 魔术号
            
        Returns:
            包含删除结果的字典
        """
        self._ensure_connected()
        
        request = trade_service_pb2.TradeRequest(
            account_id=account_id,
            action=trade_service_pb2.TRADE_ACTION_DELETE,
            symbol=symbol,
            position_id=order_id,
            magic=magic,
            timestamp=int(time.time() * 1000),
        )
        
        try:
            logger.info(f"删除挂单: 订单={order_id}")
            response = self._stub.DeleteOrder(request, timeout=self.timeout)
            
            result = {
                'success': response.success,
                'retcode': response.retcode,
                'message': response.message,
            }
            
            if response.success:
                logger.info(f"删除成功: {response.message}")
            else:
                logger.warning(f"删除失败: {response.message}")
            
            return result
            
        except grpc.RpcError as e:
            error_msg = f"gRPC 错误: {e.code()} - {e.details()}"
            logger.error(error_msg)
            return {
                'success': False,
                'retcode': -1,
                'message': error_msg,
            }
    
    def get_klines(
        self,
        symbol: str,
        timeframe: str = '1m',
        from_time: int = 0,
        to_time: int = 0,
        count: int = 0,
    ) -> Dict:
        """
        获取K线数据
        
        Args:
            symbol: 交易品种 (e.g., 'BTCUSDm')
            timeframe: 时间周期 ('1m', '5m', '1h', '1d'等)
            from_time: 开始时间（Unix时间戳，秒，0表示从最早开始）
            to_time: 结束时间（Unix时间戳，秒，0表示到最新）
            count: 数量（0表示全部）
            
        Returns:
            包含K线列表的字典:
            {
                'success': bool,
                'message': str,
                'klines': List[Dict],
                'count': int,
            }
        """
        self._ensure_connected()
        
        request = trade_service_pb2.KlineRequest(
            symbol=symbol,
            timeframe=timeframe,
            from_time=from_time,
            to_time=to_time,
            count=count,
        )
        
        try:
            logger.debug(f"查询K线: symbol={symbol}, timeframe={timeframe}, count={count}")
            response = self._stub.GetKlines(request, timeout=self.timeout)
            
            # 转换 KlineData 为字典
            klines = []
            for kline_pb in response.klines:
                klines.append({
                    'time': kline_pb.time,
                    'open': kline_pb.open,
                    'high': kline_pb.high,
                    'low': kline_pb.low,
                    'close': kline_pb.close,
                    'volume': kline_pb.volume,
                    'tick_volume': kline_pb.tick_volume,
                })
            
            result = {
                'success': response.success,
                'message': response.message,
                'klines': klines,
                'count': response.count,
            }
            
            logger.debug(f"查询K线结果: 成功={response.success}, 数量={response.count}")
            return result
            
        except grpc.RpcError as e:
            error_msg = f"gRPC 错误: {e.code()} - {e.details()}"
            logger.error(error_msg)
            return {
                'success': False,
                'message': error_msg,
                'klines': [],
                'count': 0,
            }
    
    def get_ticks(
        self,
        symbol: str,
        from_time: int = 0,
        to_time: int = 0,
        count: int = 0,
    ) -> Dict:
        """
        获取TICK历史数据
        
        Args:
            symbol: 交易品种 (e.g., 'BTCUSDm')
            from_time: 开始时间（Unix时间戳，毫秒，0表示从最早开始）
            to_time: 结束时间（Unix时间戳，毫秒，0表示到最新）
            count: 数量（0表示全部）
            
        Returns:
            包含TICK列表的字典:
            {
                'success': bool,
                'message': str,
                'ticks': List[Dict],
                'count': int,
            }
        """
        self._ensure_connected()
        
        request = trade_service_pb2.TickRequest(
            symbol=symbol,
            from_time=from_time,
            to_time=to_time,
            count=count,
        )
        
        try:
            logger.debug(f"查询TICK: symbol={symbol}, count={count}")
            response = self._stub.GetTicks(request, timeout=self.timeout)
            
            # 转换 TickData 为字典
            ticks = []
            for tick_pb in response.ticks:
                ticks.append({
                    'time_msc': tick_pb.time_msc,
                    'bid': tick_pb.bid,
                    'ask': tick_pb.ask,
                    'last': tick_pb.last,
                    'volume': tick_pb.volume,
                })
            
            result = {
                'success': response.success,
                'message': response.message,
                'ticks': ticks,
                'count': response.count,
            }
            
            logger.debug(f"查询TICK结果: 成功={response.success}, 数量={response.count}")
            return result
            
        except grpc.RpcError as e:
            error_msg = f"gRPC 错误: {e.code()} - {e.details()}"
            logger.error(error_msg)
            return {
                'success': False,
                'message': error_msg,
                'ticks': [],
                'count': 0,
            }
    
    def get_latest_tick(self, symbol: str) -> Dict:
        """
        获取最新TICK
        
        Args:
            symbol: 交易品种 (e.g., 'BTCUSDm')
            
        Returns:
            包含最新TICK的字典:
            {
                'success': bool,
                'message': str,
                'tick': Dict,
            }
        """
        self._ensure_connected()
        
        request = trade_service_pb2.LatestTickRequest(symbol=symbol)
        
        try:
            logger.debug(f"查询最新TICK: symbol={symbol}")
            response = self._stub.GetLatestTick(request, timeout=self.timeout)
            
            result = {
                'success': response.success,
                'message': response.message,
                'tick': {
                    'time_msc': response.tick.time_msc,
                    'bid': response.tick.bid,
                    'ask': response.tick.ask,
                    'last': response.tick.last,
                    'volume': response.tick.volume,
                } if response.tick else None,
            }
            
            logger.debug(f"查询最新TICK结果: 成功={response.success}")
            return result
            
        except grpc.RpcError as e:
            error_msg = f"gRPC 错误: {e.code()} - {e.details()}"
            logger.error(error_msg)
            return {
                'success': False,
                'message': error_msg,
                'tick': None,
            }
    
    def stream_ticks(self, symbol: str):
        """
        实时TICK流（服务器流式推送，客户端被动接收）
        
        工作原理：
        - 客户端建立gRPC连接（主动）
        - 服务器持续推送TICK数据流（被动接收）
        - 通过迭代器yield返回数据
        
        Args:
            symbol: 交易品种 (e.g., 'BTCUSDm')
            
        Yields:
            TICK数据字典:
            {
                'time_msc': int,
                'bid': float,
                'ask': float,
                'last': float,
                'volume': int,
            }
        """
        self._ensure_connected()
        
        request = trade_service_pb2.LatestTickRequest(symbol=symbol)
        
        try:
            logger.info(f"开始被动接收实时TICK流: symbol={symbol} (服务器推送模式)")
            # 服务器持续推送数据，客户端通过迭代器被动接收
            for tick_pb in self._stub.StreamTicks(request):
                yield {
                    'time_msc': tick_pb.time_msc,
                    'bid': tick_pb.bid,
                    'ask': tick_pb.ask,
                    'last': tick_pb.last,
                    'volume': tick_pb.volume,
                }
        except grpc.RpcError as e:
            error_msg = f"gRPC 流错误: {e.code()} - {e.details()}"
            logger.error(error_msg)
            raise
    
    def get_orders(self, account_id: str = '', symbol: str = '', ticket: int = 0, magic: int = 0) -> Dict:
        """
        获取所有订单（完整MT5结构）
        
        Args:
            account_id: 账户ID（必需）
            symbol: 交易品种（可选，空字符串表示所有）
            ticket: 订单号（可选，0表示所有）
            magic: 魔术号（可选，0表示所有）
            
        Returns:
            包含订单列表的字典:
            {
                'success': bool,
                'message': str,
                'orders': List[Dict],
                'count': int,
            }
        """
        # 🔴 修复：确保连接可用（自动重连）
        self._ensure_connected(retry_count=2)
        
        request = trade_service_pb2.QueryRequest(
            account_id=account_id,
            symbol=symbol,
            ticket=ticket,
            magic=magic,
        )
        
        try:
            logger.debug(f"查询订单: account_id={account_id}, symbol={symbol or '全部'}, ticket={ticket or '全部'}")
            response = self._stub.GetOrders(request, timeout=self.timeout)
            
            # 转换 OrderData 为字典（完整MT5标准结构）
            orders = []
            for order_pb in response.orders:
                order_dict = {
                    'ticket': order_pb.ticket,
                    'position_id': order_pb.position_id,
                    'position_by_id': order_pb.position_by_id,
                    'time_setup': order_pb.time_setup,
                    'time_setup_msc': order_pb.time_setup_msc,
                    'time_done': order_pb.time_done,
                    'time_done_msc': order_pb.time_done_msc,
                    'time_expiration': order_pb.time_expiration,
                    'type': order_pb.type,
                    'type_filling': order_pb.type_filling,
                    'type_time': order_pb.type_time,
                    'magic': order_pb.magic,
                    'state': order_pb.state,
                    'reason': order_pb.reason,
                    'volume_initial': order_pb.volume_initial,
                    'volume_current': order_pb.volume_current,
                    'price_open': order_pb.price_open,
                    'price_current': order_pb.price_current,
                    'price_stoplimit': order_pb.price_stoplimit,
                    'sl': order_pb.sl,
                    'tp': order_pb.tp,
                    'symbol': order_pb.symbol,
                    'comment': order_pb.comment,
                    'external_id': order_pb.external_id,
                }
                orders.append(order_dict)
            
            result = {
                'success': response.success,
                'message': response.message,
                'orders': orders,
                'count': response.count,
            }
            
            logger.debug(f"查询订单结果: 成功={response.success}, 数量={response.count}")
            return result
            
        except grpc.RpcError as e:
            error_msg = f"gRPC 错误: {e.code()} - {e.details()}"
            logger.error(error_msg)
            return {
                'success': False,
                'message': error_msg,
                'orders': [],
                'count': 0,
            }
    
    def get_positions(self, account_id: str = '', symbol: str = '', ticket: int = 0, magic: int = 0) -> Dict:
        """
        获取所有持仓（完整MT5结构）
        
        Args:
            account_id: 账户ID（必需）
            symbol: 交易品种（可选，空字符串表示所有）
            ticket: 持仓号（可选，0表示所有）
            magic: 魔术号（可选，0表示所有）
            
        Returns:
            包含持仓列表的字典:
            {
                'success': bool,
                'message': str,
                'positions': List[Dict],
                'count': int,
            }
        """
        # 🔴 修复：确保连接可用（自动重连）
        self._ensure_connected(retry_count=2)
        
        request = trade_service_pb2.QueryRequest(
            account_id=account_id,
            symbol=symbol,
            ticket=ticket,
            magic=magic,
        )
        
        try:
            logger.debug(f"查询持仓: account_id={account_id}, symbol={symbol or '全部'}, ticket={ticket or '全部'}")
            response = self._stub.GetPositions(request, timeout=self.timeout)
            
            # 转换 PositionData 为字典（完整MT5标准结构）
            positions = []
            for pos_pb in response.positions:
                pos_dict = {
                    'ticket': pos_pb.ticket,
                    'time': pos_pb.time,
                    'time_msc': pos_pb.time_msc,
                    'time_update': pos_pb.time_update,
                    'time_update_msc': pos_pb.time_update_msc,
                    'type': pos_pb.type,
                    'magic': pos_pb.magic,
                    'identifier': pos_pb.identifier,
                    'reason': pos_pb.reason,
                    'volume': pos_pb.volume,
                    'price_open': pos_pb.price_open,
                    'price_current': pos_pb.price_current,
                    'price_stoplimit': pos_pb.price_stoplimit,
                    'sl': pos_pb.sl,
                    'tp': pos_pb.tp,
                    'profit': pos_pb.profit,
                    'swap': pos_pb.swap,
                    'commission': pos_pb.commission,
                    'symbol': pos_pb.symbol,
                    'comment': pos_pb.comment,
                    'external_id': pos_pb.external_id,
                }
                positions.append(pos_dict)
            
            result = {
                'success': response.success,
                'message': response.message,
                'positions': positions,
                'count': response.count,
            }
            
            logger.debug(f"查询持仓结果: 成功={response.success}, 数量={response.count}")
            return result
            
        except grpc.RpcError as e:
            error_msg = f"gRPC 错误: {e.code()} - {e.details()}"
            logger.error(error_msg)
            return {
                'success': False,
                'message': error_msg,
                'positions': [],
                'count': 0,
            }
    
    def get_symbol_info(self, symbol: str) -> Dict:
        """
        获取品种信息（管理信息）
        
        Args:
            symbol: 交易品种 (e.g., 'BTCUSDm')
            
        Returns:
            包含品种信息的字典:
            {
                'success': bool,
                'message': str,
                'symbol_info': Dict,
            }
        """
        self._ensure_connected()
        
        request = trade_service_pb2.SymbolInfoRequest(symbol=symbol)
        
        try:
            logger.debug(f"查询品种信息: symbol={symbol}")
            response = self._stub.GetSymbolInfo(request, timeout=self.timeout)
            
            result = {
                'success': response.success,
                'message': response.message,
                'symbol_info': {
                    'symbol': response.symbol_info.symbol,
                    'description': response.symbol_info.description,
                    'currency_base': response.symbol_info.currency_base,
                    'currency_profit': response.symbol_info.currency_profit,
                    'currency_margin': response.symbol_info.currency_margin,
                    'digits': response.symbol_info.digits,
                    'point': response.symbol_info.point,
                    'trade_mode': response.symbol_info.trade_mode,
                    'trade_stops_level': response.symbol_info.trade_stops_level,
                    'trade_freeze_level': response.symbol_info.trade_freeze_level,
                    'volume_min': response.symbol_info.volume_min,
                    'volume_max': response.symbol_info.volume_max,
                    'volume_step': response.symbol_info.volume_step,
                    'margin_initial': response.symbol_info.margin_initial,
                    'margin_maintenance': response.symbol_info.margin_maintenance,
                    'filling_mode': response.symbol_info.filling_mode,
                    'visible': response.symbol_info.visible,
                    'select': response.symbol_info.select,
                } if response.symbol_info else None,
            }
            
            logger.debug(f"查询品种信息结果: 成功={response.success}")
            return result
            
        except grpc.RpcError as e:
            error_msg = f"gRPC 错误: {e.code()} - {e.details()}"
            logger.error(error_msg)
            return {
                'success': False,
                'message': error_msg,
                'symbol_info': None,
            }
    
    def get_account_info(self) -> Dict:
        """
        获取账户信息（管理信息）
        
        Returns:
            包含账户信息的字典:
            {
                'success': bool,
                'message': str,
                'account_info': Dict,
            }
        """
        # 🔴 修复：确保连接可用（自动重连）
        self._ensure_connected(retry_count=2)
        
        request = trade_service_pb2.AccountInfoRequest()
        
        try:
            logger.debug("查询账户信息")
            response = self._stub.GetAccountInfo(request, timeout=self.timeout)
            
            result = {
                'success': response.success,
                'message': response.message,
                'account_info': {
                    'login': response.account_info.login,
                    'server': response.account_info.server,
                    'name': response.account_info.name,
                    'company': response.account_info.company,
                    'trade_mode': response.account_info.trade_mode,
                    'leverage': response.account_info.leverage,
                    'limit_orders': response.account_info.limit_orders,
                    'balance': response.account_info.balance,
                    'credit': response.account_info.credit,
                    'profit': response.account_info.profit,
                    'equity': response.account_info.equity,
                    'margin': response.account_info.margin,
                    'margin_free': response.account_info.margin_free,
                    'margin_level': response.account_info.margin_level,
                    'margin_so_call': response.account_info.margin_so_call,
                    'margin_so_so': response.account_info.margin_so_so,
                    'currency': response.account_info.currency,
                    'trade_allowed': response.account_info.trade_allowed,
                    'trade_expert': response.account_info.trade_expert,
                    'ping_last': response.account_info.ping_last,
                } if response.account_info else None,
            }
            
            logger.debug(f"查询账户信息结果: 成功={response.success}")
            return result
            
        except grpc.RpcError as e:
            error_msg = f"gRPC 错误: {e.code()} - {e.details()}"
            logger.error(error_msg)
            return {
                'success': False,
                'message': error_msg,
                'account_info': None,
            }
    
    def get_terminal_info(self) -> Dict:
        """
        获取终端信息（管理信息）
        
        Returns:
            包含终端信息的字典:
            {
                'success': bool,
                'message': str,
                'terminal_info': Dict,
            }
        """
        self._ensure_connected()
        
        request = trade_service_pb2.TerminalInfoRequest()
        
        try:
            logger.debug("查询终端信息")
            response = self._stub.GetTerminalInfo(request, timeout=self.timeout)
            
            result = {
                'success': response.success,
                'message': response.message,
                'terminal_info': {
                    'company': response.terminal_info.company,
                    'name': response.terminal_info.name,
                    'path': response.terminal_info.path,
                    'data_path': response.terminal_info.data_path,
                    'common_path': response.terminal_info.common_path,
                    'build': response.terminal_info.build,
                    'max_bars': response.terminal_info.max_bars,
                    'codepage': response.terminal_info.codepage,
                    'ping_last': response.terminal_info.ping_last,
                    'community_account': response.terminal_info.community_account,
                    'community_connection': response.terminal_info.community_connection,
                    'connected': response.terminal_info.connected,
                    'dlls_allowed': response.terminal_info.dlls_allowed,
                    'trade_allowed': response.terminal_info.trade_allowed,
                    'tradeapi_disabled': response.terminal_info.tradeapi_disabled,
                    'email_enabled': response.terminal_info.email_enabled,
                    'ftp_enabled': response.terminal_info.ftp_enabled,
                    'notifications_enabled': response.terminal_info.notifications_enabled,
                    'mqid': response.terminal_info.mqid,
                    'max_orders': response.terminal_info.max_orders,
                } if response.terminal_info else None,
            }
            
            logger.debug(f"查询终端信息结果: 成功={response.success}")
            return result
            
        except grpc.RpcError as e:
            error_msg = f"gRPC 错误: {e.code()} - {e.details()}"
            logger.error(error_msg)
            return {
                'success': False,
                'message': error_msg,
                'terminal_info': None,
            }
    
    def get_server_time(self) -> Dict:
        """
        获取服务器时间（管理信息）
        
        Returns:
            包含服务器时间的字典:
            {
                'success': bool,
                'message': str,
                'time': int,      # Unix时间戳（秒）
                'time_msc': int,  # Unix时间戳（毫秒）
            }
        """
        self._ensure_connected()
        
        request = trade_service_pb2.ServerTimeRequest()
        
        try:
            logger.debug("查询服务器时间")
            response = self._stub.GetServerTime(request, timeout=self.timeout)
            
            result = {
                'success': response.success,
                'message': response.message,
                'time': response.time,
                'time_msc': response.time_msc,
            }
            
            logger.debug(f"查询服务器时间结果: 成功={response.success}, time={response.time}")
            return result
            
        except grpc.RpcError as e:
            error_msg = f"gRPC 错误: {e.code()} - {e.details()}"
            logger.error(error_msg)
            return {
                'success': False,
                'message': error_msg,
                'time': 0,
                'time_msc': 0,
            }
    
    def get_symbol_list(self, group: str = '') -> Dict:
        """
        获取品种列表（管理信息）
        
        Args:
            group: 品种组（可选，空字符串表示所有）
            
        Returns:
            包含品种列表的字典:
            {
                'success': bool,
                'message': str,
                'symbols': List[str],
                'count': int,
            }
        """
        self._ensure_connected()
        
        request = trade_service_pb2.SymbolListRequest(group=group)
        
        try:
            logger.debug(f"查询品种列表: group={group or '全部'}")
            response = self._stub.GetSymbolList(request, timeout=self.timeout)
            
            result = {
                'success': response.success,
                'message': response.message,
                'symbols': list(response.symbols),
                'count': response.count,
            }
            
            logger.debug(f"查询品种列表结果: 成功={response.success}, 数量={response.count}")
            return result
            
        except grpc.RpcError as e:
            error_msg = f"gRPC 错误: {e.code()} - {e.details()}"
            logger.error(error_msg)
            return {
                'success': False,
                'message': error_msg,
                'symbols': [],
                'count': 0,
            }
    
    def close(self):
        """关闭 gRPC 连接"""
        if self._channel:
            self._channel.close()
            self._channel = None
            self._stub = None
            logger.debug("已关闭 gRPC 连接")
    
    def __enter__(self):
        """上下文管理器入口"""
        self._ensure_connected()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()


# 全局客户端实例（单例模式）
_global_client: Optional[GrpcTradeClient] = None


def is_grpc_available() -> bool:
    """
    检查 gRPC 功能是否可用
    
    Returns:
        bool: 如果 gRPC 代码已生成且可用，返回 True
    """
    return _grpc_available


def get_grpc_client(host: str = None, port: int = None) -> GrpcTradeClient:
    """
    获取全局 gRPC 客户端实例（单例模式）
    
    Args:
        host: Windows 主机地址
        port: gRPC 端口
        
    Returns:
        GrpcTradeClient 实例
    """
    global _global_client
    
    if _global_client is None:
        _global_client = GrpcTradeClient(host=host, port=port)
    
    return _global_client

