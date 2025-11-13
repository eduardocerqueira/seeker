#date: 2025-11-13T17:03:12Z
#url: https://api.github.com/gists/d68fdd03a1f1096971a5f4ed71530eb2
#owner: https://api.github.com/users/wangwei334455

"""
MT5连接池 - 单例模式
确保整个进程中只有一个MT5连接实例，所有采集器共享
"""
import threading
import logging
import time
from typing import Optional, Dict
from datetime import datetime
import MetaTrader5 as mt5

from mt5_error_handler import MT5ErrorHandler


class MT5ConnectionPool:
    """MT5连接池（单例模式）"""
    
    _instance: Optional['MT5ConnectionPool'] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """确保单例"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """初始化连接池"""
        # 避免重复初始化
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        self.logger = logging.getLogger("MT5ConnectionPool")
        
        # MT5连接状态
        self.connected = False
        self.connection_lock = threading.Lock()
        self.config = None
        
        # 使用计数（用于追踪有多少采集器在使用）
        self.usage_count = 0
        self.usage_lock = threading.Lock()
        
        # 健康检查和重连
        self.last_health_check = None
        self.consecutive_errors = 0
        self.max_consecutive_errors = 10
        self.error_handler = MT5ErrorHandler()
    
    def connect(self, config: dict) -> bool:
        """
        连接MT5（多个采集器可以安全调用）
        
        Args:
            config: MT5配置字典
            
        Returns:
            bool: 是否连接成功
        """
        with self.connection_lock:
            # 如果已连接，增加使用计数
            if self.connected:
                with self.usage_lock:
                    self.usage_count += 1
                self.logger.info(f"MT5已连接，当前使用数: {self.usage_count}")
                return True
            
            # 首次连接
            self.logger.info("首次连接MT5...")
            self.config = config
            
            try:
                # 初始化MT5
                if not mt5.initialize(
                    login=config['login'],
                    password= "**********"
                    server=config['server'],
                    timeout=config.get('timeout', 60000)
                ):
                    # 使用详细错误处理
                    error_code, error_msg, error_name, solution = self.error_handler.get_last_error()
                    self.logger.error(f"MT5初始化失败 [{error_code}]: {error_name}")
                    self.logger.error(f"  原始消息: {error_msg}")
                    self.logger.error(f"  解决方案: {solution}")
                    return False
                
                # 记录MT5 Python库版本
                version = mt5.version()
                self.logger.info(f"MetaTrader5 Python库版本: {version}")
                
                # 检查终端信息
                terminal_info = mt5.terminal_info()
                if terminal_info is None:
                    self.logger.error("无法获取终端信息")
                    self.error_handler.log_error(self.logger, "获取终端信息时")
                    mt5.shutdown()
                    return False
                
                # 验证终端状态
                if not terminal_info.connected:
                    self.logger.error("MT5终端未连接到交易服务器")
                    mt5.shutdown()
                    return False
                
                if terminal_info.tradeapi_disabled:
                    self.logger.error("MT5终端禁用了交易API")
                    mt5.shutdown()
                    return False
                
                self.logger.info(f"✓ MT5终端信息:")
                self.logger.info(f"  终端名称: {terminal_info.name}")
                self.logger.info(f"  编译版本: {terminal_info.build}")
                self.logger.info(f"  公司: {terminal_info.company}")
                self.logger.info(f"  服务器连接: {'是' if terminal_info.connected else '否'}")
                self.logger.info(f"  交易允许: {'是' if terminal_info.trade_allowed else '否'}")
                self.logger.info(f"  最近ping: {terminal_info.ping_last}ms")
                
                # 验证连接
                account_info = mt5.account_info()
                if account_info is None:
                    self.logger.error("无法获取账户信息")
                    self.error_handler.log_error(self.logger, "获取账户信息时")
                    mt5.shutdown()
                    return False
                
                # 记录详细账户信息
                self.logger.info(f"✓ MT5账户信息:")
                self.logger.info(f"  账号: {account_info.login}")
                self.logger.info(f"  服务器: {account_info.server}")
                self.logger.info(f"  账户类型: {'模拟' if account_info.trade_mode == 0 else '真实'}")
                self.logger.info(f"  杠杆: 1:{account_info.leverage}")
                self.logger.info(f"  货币: {account_info.currency}")
                self.logger.info(f"  余额: {account_info.balance}")
                self.logger.info(f"  净值: {account_info.equity}")
                self.logger.info(f"  可用保证金: {account_info.margin_free}")
                self.logger.info(f"  交易允许: {'是' if account_info.trade_allowed else '否'}")
                self.logger.info(f"  EA交易允许: {'是' if account_info.trade_expert else '否'}")
                
                self.connected = True
                self.consecutive_errors = 0  # 重置错误计数
                with self.usage_lock:
                    self.usage_count = 1
                
                # 记录首次健康检查时间
                self.last_health_check = time.time()
                
                return True
                
            except Exception as e:
                self.logger.error(f"MT5连接异常: {e}")
                return False
    
    def disconnect(self, force: bool = False):
        """
        断开MT5连接（只有当没有采集器使用时才真正断开）
        
        Args:
            force: 强制断开，不管使用计数
        """
        with self.connection_lock:
            if not self.connected:
                return
            
            with self.usage_lock:
                self.usage_count = max(0, self.usage_count - 1)
                
                # 只有当使用计数为0或强制断开时才真正断开
                if self.usage_count > 0 and not force:
                    self.logger.info(f"MT5仍有使用者，当前使用数: {self.usage_count}")
                    return
            
            # 真正断开连接
            self.logger.info("断开MT5连接...")
            try:
                mt5.shutdown()
                self.connected = False
                self.logger.info("✓ MT5连接已断开")
            except Exception as e:
                self.logger.error(f"断开MT5连接异常: {e}")
    
    def get_tick(self, symbol: str) -> Optional[dict]:
        """
        获取TICK数据（线程安全）
        
        Args:
            symbol: 交易品种
            
        Returns:
            dict: TICK数据字典
        """
        if not self.connected:
            return None
        
        try:
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return None
            
            return {
                'time': tick.time,
                'bid': tick.bid,
                'ask': tick.ask,
                'last': tick.last,
                'volume': tick.volume,
                'time_msc': tick.time_msc,
                'flags': tick.flags,
                'volume_real': tick.volume_real,
            }
        except Exception as e:
            self.logger.error(f"获取TICK数据异常: {e}")
            return None
    
    def get_klines(self, symbol: str, timeframe: int, count: int, max_batch_size: int = 50000) -> Optional[list]:
        """
        获取K线数据（线程安全，支持大批量分批获取）
        
        Args:
            symbol: 交易品种
            timeframe: 时间周期
            count: K线数量
            max_batch_size: 单次最大获取数量（默认50000，避免MT5服务器限制）
            
        Returns:
            list: K线数据列表
        """
        if not self.connected:
            return None
        
        try:
            # 如果请求量不大，直接一次获取
            if count <= max_batch_size:
                rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
                if rates is None or len(rates) == 0:
                    return None
                
                return self._convert_rates_to_klines(rates)
            
            # 大批量数据：分批获取
            self.logger.info(f"大批量请求 {count} 条K线，将分 {(count + max_batch_size - 1) // max_batch_size} 批获取...")
            
            all_klines = []
            remaining = count
            offset = 0
            
            while remaining > 0:
                batch_size = min(remaining, max_batch_size)
                
                rates = mt5.copy_rates_from_pos(symbol, timeframe, offset, batch_size)
                
                if rates is None or len(rates) == 0:
                    if offset == 0:
                        # 第一批就失败
                        self.logger.error(f"获取K线数据失败（offset={offset}）")
                        return None
                    else:
                        # 后续批次失败，返回已获取的数据
                        self.logger.warning(f"批次 {offset} 获取失败，返回已获取的 {len(all_klines)} 条")
                        break
                
                klines = self._convert_rates_to_klines(rates)
                all_klines.extend(klines)
                
                offset += len(rates)
                remaining -= len(rates)
                
                self.logger.info(f"已获取 {len(all_klines)}/{count} 条K线...")
                
                # 如果返回的数量少于请求的，说明已经到头了
                if len(rates) < batch_size:
                    self.logger.info(f"MT5服务器只提供 {len(all_klines)} 条历史数据")
                    break
            
            return all_klines
            
        except Exception as e:
            self.logger.error(f"获取K线数据异常: {e}")
            return None
    
    def _convert_rates_to_klines(self, rates) -> list:
        """
        将MT5的rates数据转换为标准K线格式
        
        Args:
            rates: MT5返回的rates数组
            
        Returns:
            list: K线数据列表
        """
        klines = []
        for rate in rates:
            klines.append({
                'time': int(rate['time']),
                'open': float(rate['open']),
                'high': float(rate['high']),
                'low': float(rate['low']),
                'close': float(rate['close']),
                'volume': int(rate['tick_volume']),
                'real_volume': int(rate['real_volume']) if 'real_volume' in rate.dtype.names else 0,
            })
        return klines
    
    def get_klines_by_date_range(self, symbol: str, timeframe: int, 
                                 date_from: datetime, date_to: datetime) -> Optional[list]:
        """
        按日期范围获取K线数据（官方推荐方法）
        
        这个方法比copy_rates_from_pos更可靠，因为它使用明确的时间范围
        
        Args:
            symbol: 交易品种
            timeframe: 时间周期
            date_from: 开始时间
            date_to: 结束时间
            
        Returns:
            list: K线数据列表
        """
        if not self.connected:
            return None
        
        try:
            rates = mt5.copy_rates_range(symbol, timeframe, date_from, date_to)
            
            if rates is None or len(rates) == 0:
                error_code, error_msg, error_name, solution = self.error_handler.get_last_error()
                self.logger.error(f"按日期范围获取K线失败 [{error_code}]: {error_name}")
                self.logger.error(f"  时间范围: {date_from} -> {date_to}")
                self.logger.error(f"  解决方案: {solution}")
                return None
            
            self.logger.info(f"按日期范围获取K线成功: {len(rates)} 条 ({date_from} -> {date_to})")
            return self._convert_rates_to_klines(rates)
            
        except Exception as e:
            self.logger.error(f"按日期范围获取K线异常: {e}")
            return None
    
    def check_symbol(self, symbol: str, detailed: bool = False) -> bool:
        """
        检查交易品种是否可用
        
        Args:
            symbol: 交易品种
            detailed: 是否输出详细信息
            
        Returns:
            bool: 是否可用
        """
        if not self.connected:
            return False
        
        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                self.logger.error(f"交易品种 {symbol} 不存在")
                self.error_handler.log_error(self.logger, f"获取品种 {symbol} 信息时")
                return False
            
            # 如果品种不可见，尝试选择它
            if not symbol_info.visible:
                self.logger.info(f"品种 {symbol} 不可见，尝试启用...")
                if not mt5.symbol_select(symbol, True):
                    self.logger.error(f"无法启用品种 {symbol}")
                    return False
                
                # 再次检查是否可见
                symbol_info = mt5.symbol_info(symbol)
                if not symbol_info.visible:
                    self.logger.error(f"品种 {symbol} 启用后仍不可见")
                    return False
            
            # 检查交易模式
            if symbol_info.trade_mode == mt5.SYMBOL_TRADE_MODE_DISABLED:
                self.logger.error(f"品种 {symbol} 禁止交易")
                return False
            
            # 输出详细信息
            if detailed:
                self.logger.info(f"✓ 交易品种 {symbol} 详细信息:")
                self.logger.info(f"  可见性: {'是' if symbol_info.visible else '否'}")
                self.logger.info(f"  交易模式: {self._get_trade_mode_name(symbol_info.trade_mode)}")
                self.logger.info(f"  小数位数: {symbol_info.digits}")
                self.logger.info(f"  当前点差: {symbol_info.spread}")
                self.logger.info(f"  浮动点差: {'是' if symbol_info.spread_float else '否'}")
                self.logger.info(f"  当前会话成交数: {symbol_info.session_deals}")
                self.logger.info(f"  合约大小: {symbol_info.trade_contract_size}")
                self.logger.info(f"  最小手数: {symbol_info.volume_min}")
                self.logger.info(f"  最大手数: {symbol_info.volume_max}")
                self.logger.info(f"  手数步进: {symbol_info.volume_step}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"检查交易品种异常: {e}")
            return False
    
    def _get_trade_mode_name(self, trade_mode: int) -> str:
        """获取交易模式名称"""
        modes = {
            0: "禁止交易",
            1: "仅平仓",
            2: "完全交易",
            3: "仅做多",
            4: "仅做空",
        }
        return modes.get(trade_mode, f"未知模式({trade_mode})")
    
    def get_usage_count(self) -> int:
        """获取当前使用数"""
        with self.usage_lock:
            return self.usage_count
    
    def is_connected(self) -> bool:
        """检查是否已连接"""
        return self.connected
    
    def check_connection_health(self) -> bool:
        """
        检查MT5连接健康状态
        
        Returns:
            bool: 连接是否健康
        """
        if not self.connected:
            return False
        
        try:
            # 尝试获取账户信息作为心跳
            account_info = mt5.account_info()
            if account_info is None:
                self.logger.warning("健康检查失败: 无法获取账户信息")
                self.error_handler.log_error(self.logger, "健康检查时")
                self.consecutive_errors += 1
                return False
            
            # 健康检查成功
            self.last_health_check = time.time()
            self.consecutive_errors = 0
            self.logger.debug(f"健康检查通过 - 账户: {account_info.login}, 余额: {account_info.balance}")
            return True
            
        except Exception as e:
            self.logger.error(f"健康检查异常: {e}")
            self.consecutive_errors += 1
            return False
    
    def reconnect(self, max_retries: int = 3) -> bool:
        """
        自动重连MT5
        
        Args:
            max_retries: 最大重试次数
            
        Returns:
            bool: 是否重连成功
        """
        if not self.config:
            self.logger.error("无法重连: 缺少配置信息")
            return False
        
        self.logger.info(f"开始自动重连MT5（最多 {max_retries} 次）...")
        
        # 先断开旧连接
        try:
            mt5.shutdown()
        except:
            pass
        
        self.connected = False
        
        # 重试连接
        for i in range(max_retries):
            self.logger.info(f"重连尝试 {i+1}/{max_retries}...")
            
            if self.connect(self.config):
                self.logger.info(f"✓ 重连成功（第 {i+1} 次尝试）")
                return True
            
            if i < max_retries - 1:
                wait_time = 5 * (i + 1)  # 递增等待时间
                self.logger.info(f"等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
        
        self.logger.error(f"❌ 重连失败（已尝试 {max_retries} 次）")
        return False
    
    def check_time_sync(self, symbol: str = "BTCUSDm") -> Dict[str, any]:
        """
        检查本地时间与MT5服务器时间同步
        
        Args:
            symbol: 用于获取服务器时间的交易品种
            
        Returns:
            dict: 时间同步信息
        """
        if not self.connected:
            return {'synced': False, 'error': 'Not connected'}
        
        try:
            # 获取服务器时间
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return {'synced': False, 'error': 'Failed to get server time'}
            
            server_time = tick.time
            local_time = int(time.time())
            diff = abs(server_time - local_time)
            
            # 允许最大3秒误差
            synced = diff <= 3
            
            result = {
                'synced': synced,
                'server_time': datetime.fromtimestamp(server_time),
                'local_time': datetime.fromtimestamp(local_time),
                'diff_seconds': diff
            }
            
            if not synced:
                self.logger.warning(f"时间不同步: 本地与服务器相差 {diff} 秒")
                self.logger.warning(f"  服务器时间: {result['server_time']}")
                self.logger.warning(f"  本地时间: {result['local_time']}")
            else:
                self.logger.debug(f"时间同步良好，误差 {diff} 秒")
            
            return result
            
        except Exception as e:
            self.logger.error(f"时间同步检查异常: {e}")
            return {'synced': False, 'error': str(e)}
    
    def validate_kline_quality(self, kline: dict) -> bool:
        """
        验证K线数据质量
        
        Args:
            kline: K线数据字典
            
        Returns:
            bool: 数据是否有效
        """
        try:
            # 1. 检查必需字段
            required_fields = ['open', 'high', 'low', 'close', 'volume']
            if not all(field in kline for field in required_fields):
                self.logger.warning(f"K线缺少必需字段: {kline}")
                return False
            
            # 2. 检查OHLC关系
            if not (kline['low'] <= kline['open'] <= kline['high'] and \
                    kline['low'] <= kline['close'] <= kline['high']):
                self.logger.warning(f"OHLC关系异常: {kline}")
                return False
            
            # 3. 检查价格为0或负数
            if any(kline[k] <= 0 for k in ['open', 'high', 'low', 'close']):
                self.logger.warning(f"价格异常（<=0）: {kline}")
                return False
            
            # 4. 检查成交量
            if kline['volume'] < 0:
                self.logger.warning(f"成交量异常（<0）: {kline}")
                return False
            
            # 5. 检查价格跳变过大（超过50%认为异常）
            if kline['high'] / kline['low'] > 1.5:
                self.logger.warning(f"价格波动异常（>50%）: {kline}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"K线质量验证异常: {e}")
            return False


# 全局连接池实例
_mt5_pool = MT5ConnectionPool()


def get_mt5_pool() -> MT5ConnectionPool:
    """获取全局MT5连接池实例"""
    return _mt5_pool

pool

