#date: 2025-11-13T17:03:06Z
#url: https://api.github.com/gists/8dde3cf7f589cb5afb9a45bd262da41a
#owner: https://api.github.com/users/wangwei334455

"""
L2核心决策层 - 配置管理器
负责从Redis加载并监听策略参数，实现前端热更新
"""
import redis
import json
import threading
from threading import Thread, Event
from typing import Dict, Any, Optional
from loguru import logger

# 导入默认配置
try:
    from config.default_configs import (
        CONFIG_GLOBAL,
        CONFIG_RANGING,
        CONFIG_UPTREND,
        CONFIG_DOWNTREND,
        CONFIG_RANGES,
        init_all_configs
    )
except ImportError:
    # 如果导入失败，使用硬编码默认值
    CONFIG_GLOBAL = {'KLINE_PERIOD_MIN': '1', 'HISTORY_CANDLES_N': '20', 'MA_SHORT_PERIOD': '5', 'MA_LONG_PERIOD': '20'}
    CONFIG_RANGING = {'BBANDS_SD_MULTIPLIER': '2.0', 'ADX_MAX_THRESHOLD': '25.0', 'LRS_REVERSE_THRESHOLD': '0.00005'}
    CONFIG_UPTREND = {'ADX_MIN_THRESHOLD': '30.0', 'LRS_MIN_MOMENTUM': '0.00015'}
    CONFIG_DOWNTREND = {'ADX_MIN_THRESHOLD': '30.0', 'LRS_MIN_MOMENTUM': '0.00015'}
    CONFIG_RANGES = {}
    def init_all_configs(redis_client):
        pass


class ConfigManager:
    """
    L2配置管理器
    
    职责：
    1. 从Redis加载配置（如果不存在则初始化默认值）
    2. 监听Redis Pub/Sub通道，实现配置热更新
    3. 提供线程安全的配置访问接口
    4. 配置验证和类型转换
    """
    
    def __init__(self, 
                 redis_host='localhost', 
                 redis_port=6379, 
                 redis_db=0,
                 channel='CONFIG:UPDATE'):
        """
        初始化配置管理器
        
        Args:
            redis_host: Redis主机地址
            redis_port: Redis端口
            redis_db: Redis数据库编号
            channel: Pub/Sub通道名称
        """
        # Redis连接（decode_responses=True，自动解码字符串）
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            decode_responses=True  # 自动解码为字符串
        )
        
        # 测试连接
        try:
            self.redis_client.ping()
            logger.info(f"ConfigManager: Redis连接成功 {redis_host}:{redis_port}")
        except Exception as e:
            logger.error(f"ConfigManager: Redis连接失败: {e}")
            raise
        
        # Pub/Sub订阅
        self.pubsub = self.redis_client.pubsub()
        self.channel = channel
        self.pubsub.subscribe(channel)
        
        # 内存中的活跃配置（按模式分类）
        self.configs = {
            'GLOBAL': {},
            'RANGING': {},
            'UPTREND': {},
            'DOWNTREND': {},
        }
        
        # 配置锁（保证原子性更新）
        self.config_lock = threading.Lock()
        
        # 停止事件
        self._stop_event = Event()
        
        # 1. 初始化或加载配置
        self._initialize_or_load_config()
        
        # 2. 启动监听线程
        self.listener_thread = Thread(target=self._config_listener, daemon=True, name="ConfigListener")
        self.listener_thread.start()
        logger.info("ConfigManager: 配置监听线程已启动")
    
    def _initialize_or_load_config(self):
        """
        加载Redis配置到内存。如果Redis中无配置，则写入默认配置。
        
        原子性操作：使用锁保证配置更新的原子性
        """
        try:
            # 初始化所有配置到Redis（如果不存在）
            init_all_configs(self.redis_client)
            
            # 原子性加载所有模式的配置
            with self.config_lock:
                # 加载全局配置
                self._load_config_from_redis('GLOBAL', CONFIG_GLOBAL)
                
                # 加载震荡模式配置
                self._load_config_from_redis('RANGING', CONFIG_RANGING)
                
                # 加载上涨趋势模式配置
                self._load_config_from_redis('UPTREND', CONFIG_UPTREND)
                
                # 加载下跌趋势模式配置
                self._load_config_from_redis('DOWNTREND', CONFIG_DOWNTREND)
                
            logger.info("ConfigManager: 所有配置加载完成")
            
        except Exception as e:
            logger.error(f"ConfigManager: 配置加载失败，使用硬编码默认值: {e}")
            # Fallback到默认配置
            with self.config_lock:
                self.configs = {
                    'GLOBAL': self._parse_config_dict(CONFIG_GLOBAL),
                    'RANGING': self._parse_config_dict(CONFIG_RANGING),
                    'UPTREND': self._parse_config_dict(CONFIG_UPTREND),
                    'DOWNTREND': self._parse_config_dict(CONFIG_DOWNTREND),
                }
    
    def _load_config_from_redis(self, mode: str, default_config: Dict[str, str]):
        """
        从Redis加载指定模式的配置
        
        Args:
            mode: 配置模式（'GLOBAL', 'RANGING', 'UPTREND', 'DOWNTREND'）
            default_config: 默认配置字典（字符串格式）
        """
        redis_key = f'CONFIG:{mode}'
        
        # 从Redis Hash加载
        current_config = self.redis_client.hgetall(redis_key)
        
        if not current_config:
            # 如果Redis中没有，则初始化为默认配置
            self.redis_client.hset(redis_key, mapping=default_config)
            parsed_config = self._parse_config_dict(default_config)
            self.configs[mode] = parsed_config
            logger.info(f"ConfigManager: 初始化并加载默认配置 for {mode}")
        else:
            # 解析配置（类型转换）
            parsed_config = self._parse_config_dict(current_config)
            
            # 验证配置
            if self._validate_config(mode, parsed_config):
                self.configs[mode] = parsed_config
                logger.info(f"ConfigManager: 加载Redis配置 for {mode}")
            else:
                # 验证失败，使用默认配置
                logger.warning(f"ConfigManager: {mode}配置验证失败，使用默认配置")
                parsed_config = self._parse_config_dict(default_config)
                self.configs[mode] = parsed_config
    
    def _parse_config_dict(self, config_dict: Dict[str, str]) -> Dict[str, Any]:
        """
        解析配置字典，将字符串转换为正确的类型
        
        Args:
            config_dict: 配置字典（值都是字符串）
            
        Returns:
            解析后的配置字典（正确的类型）
        """
        parsed = {}
        for key, value in config_dict.items():
            # 尝试转换为数值类型
            try:
                # 检查是否为浮点数（包含小数点或科学计数法）
                if '.' in str(value) or 'e' in str(value).lower() or 'E' in str(value):
                    parsed[key] = float(value)
                else:
                    # 尝试整数
                    parsed[key] = int(value)
            except (ValueError, TypeError):
                # 转换失败，保持字符串
                parsed[key] = value
        
        return parsed
    
    def _validate_config(self, mode: str, config: Dict[str, Any]) -> bool:
        """
        验证配置的有效性
        
        Args:
            mode: 配置模式
            config: 配置字典
            
        Returns:
            bool: 验证是否通过
        """
        if mode not in CONFIG_RANGES:
            return True  # 没有定义范围，跳过验证
        
        ranges = CONFIG_RANGES[mode]
        
        for key, value in config.items():
            if key not in ranges:
                continue  # 没有定义范围，跳过
            
            min_val, max_val = ranges[key]
            
            # 类型检查
            if not isinstance(value, (int, float)):
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    logger.error(f"ConfigManager: 参数 {key} 类型错误: {value}")
                    return False
            
            # 范围检查
            if not (min_val <= value <= max_val):
                logger.error(f"ConfigManager: 参数 {key} 值 {value} 超出范围 [{min_val}, {max_val}]")
                return False
        
        return True
    
    def _config_listener(self):
        """
        后台线程：监听Redis Pub/Sub频道，实时更新配置
        
        当收到配置更新通知时，重新加载所有配置
        """
        logger.info("ConfigManager: 配置监听线程已启动")
        
        try:
            while not self._stop_event.is_set():
                try:
                    # 🔴 修复：使用get_message()替代listen()，避免连接关闭时的I/O错误
                    message = self.pubsub.get_message(timeout=1.0)
                    
                    if message is None:
                        continue
                    
                    if message['type'] == 'message':
                        # 收到通知，立即重新加载所有配置
                        data = message['data']
                        logger.info(f"ConfigManager: 收到配置更新通知: {data}")
                        
                        try:
                            # 重新加载所有配置（原子操作）
                            self._initialize_or_load_config()
                            logger.info("ConfigManager: 配置已更新")
                        except Exception as e:
                            logger.error(f"ConfigManager: 配置更新失败: {e}")
                    elif message['type'] == 'subscribe':
                        logger.debug(f"ConfigManager: 已订阅频道: {message['channel']}")
                    elif message['type'] == 'unsubscribe':
                        logger.debug(f"ConfigManager: 已取消订阅频道: {message['channel']}")
                        
                except redis.exceptions.ConnectionError as e:
                    logger.error(f"ConfigManager: Redis连接错误: {e}，等待重连...")
                    self._stop_event.wait(5)  # 等待5秒后重试
                    # 尝试重新连接
                    try:
                        self.redis_client.ping()
                        logger.info("ConfigManager: Redis连接已恢复")
                    except:
                        logger.warning("ConfigManager: Redis连接仍未恢复，继续等待...")
                except ValueError as e:
                    # 🔴 修复：处理"I/O operation on closed file"错误
                    if "closed file" in str(e) or "I/O operation" in str(e):
                        logger.warning(f"ConfigManager: Redis连接已关闭，停止监听: {e}")
                        break
                    else:
                        raise
                except Exception as e:
                    logger.error(f"ConfigManager: 监听异常: {e}")
                    self._stop_event.wait(1)  # 等待1秒后继续
                    
        except Exception as e:
            logger.error(f"ConfigManager: 监听线程异常退出: {e}")
        finally:
            logger.info("ConfigManager: 配置监听线程已停止")
    
    def get(self, mode: str, key: str, default=None):
        """
        获取配置参数值（线程安全）
        
        Args:
            mode: 配置模式（'GLOBAL', 'RANGING', 'UPTREND', 'DOWNTREND'）
            key: 参数名称
            default: 默认值（如果不存在）
            
        Returns:
            配置值
        """
        with self.config_lock:
            return self.configs.get(mode, {}).get(key, default)
    
    def get_all(self, mode: str) -> Dict[str, Any]:
        """
        获取指定模式的所有配置（线程安全）
        
        Args:
            mode: 配置模式
            
        Returns:
            配置字典
        """
        with self.config_lock:
            return self.configs.get(mode, {}).copy()
    
    def stop(self):
        """停止监听线程"""
        self._stop_event.set()
        self.pubsub.unsubscribe(self.channel)
        self.pubsub.close()
        logger.info("ConfigManager: 监听线程已停止")


# ==================== 演示和测试代码 ====================

def simulate_frontend_update(redis_client, new_lrs_threshold: float):
    """
    L3监控层模拟配置修改，并发送通知
    
    Args:
        redis_client: Redis客户端
        new_lrs_threshold: 新的LRS阈值
    """
    # 1. 修改配置（L3写入Redis Hash）
    redis_client.hset('CONFIG:RANGING', 'LRS_REVERSE_THRESHOLD', str(new_lrs_threshold))
    
    # 2. 发送通知（L3 Pub/Sub）
    redis_client.publish('CONFIG:UPDATE', f'RANGING LRS updated to {new_lrs_threshold}')
    logger.info(f"Frontend Simulator: 已修改LRS为 {new_lrs_threshold} 并发送通知")


if __name__ == '__main__':
    import time
    
    # 启动配置管理器
    logger.info("=" * 60)
    logger.info("启动ConfigManager测试")
    logger.info("=" * 60)
    
    manager = ConfigManager()
    
    # 演示L2核心线程如何读取配置
    lrs_threshold = manager.get('RANGING', 'LRS_REVERSE_THRESHOLD')
    logger.info(f"[L2 FSM Thread] 当前震荡LRS阈值: {lrs_threshold}")
    
    # 等待一下，确保监听线程启动
    time.sleep(1)
    
    # 模拟前端修改配置
    redis_test = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    simulate_frontend_update(redis_test, 0.0001)
    
    # 等待监听线程捕获并更新
    time.sleep(2)
    
    # 演示L2核心线程读取新配置
    new_lrs_threshold = manager.get('RANGING', 'LRS_REVERSE_THRESHOLD')
    logger.info(f"[L2 FSM Thread] 更新后震荡LRS阈值: {new_lrs_threshold}")
    
    # 显示所有配置
    logger.info("\n当前所有配置:")
    for mode in ['GLOBAL', 'RANGING', 'UPTREND', 'DOWNTREND']:
        config = manager.get_all(mode)
        logger.info(f"\n{mode}配置:")
        for key, value in config.items():
            logger.info(f"  {key}: {value}")
    
    time.sleep(1)
    manager.stop()
    logger.info("测试完成")

