#date: 2025-11-13T17:02:54Z
#url: https://api.github.com/gists/4c793df7bd6fb44895f49714b67dd982
#owner: https://api.github.com/users/wangwei334455

"""
独立的K线构建服务

【职责】
1. 监听已验证的TICK流，构建K线
2. 存储历史K线到Redis
3. 推送当前K线到Redis Pub/Sub（供前端实时显示）

【架构设计】
- 独立服务，不依赖策略服务
- 前端可以独立获取K线数据，无需策略服务运行
- 符合单一职责原则（SRP）
"""
import json
import time
import threading
from typing import Dict, Any, Optional
from loguru import logger
import redis

from config.redis_config import REDIS_CONFIG, REDIS_KEYS
from src.trading.core.kline_builder_enhanced import KlineBuilder as KlineBuilderEnhanced


class KlineService:
    """
    独立的K线构建服务
    
    【数据流】
    输入: Redis Stream (tick:{symbol}:validated:stream) - 已验证的TICK流
    输出: 
      - Redis Sorted Set (kline:{symbol}:1m) - 历史K线
      - Redis Pub/Sub (current_kline:{symbol}:m1) - 当前K线（实时跳动）
    
    【职责】
    - 构建K线（使用KlineBuilder）
    - 存储历史K线
    - 推送当前K线（实时更新）
    """
    
    def __init__(self, symbol: str = "BTCUSDm"):
        """
        初始化K线服务
        
        Args:
            symbol: 交易品种
        """
        self.symbol = symbol
        self.stop_event = threading.Event()
        
        # Redis连接
        self.redis_client = redis.Redis(
            host=REDIS_CONFIG.get('host', 'localhost'),
            port=REDIS_CONFIG.get('port', 6379),
            db=REDIS_CONFIG.get('db', 0),
            decode_responses=False  # 二进制模式，用于读取Stream
        )
        self.r_text = redis.Redis(
            host=REDIS_CONFIG.get('host', 'localhost'),
            port=REDIS_CONFIG.get('port', 6379),
            db=REDIS_CONFIG.get('db', 0),
            decode_responses=True  # 文本模式，用于Pub/Sub和存储
        )
        
        # Redis Keys
        self.validated_tick_stream_key = f"tick:{symbol}:validated:stream"  # 已验证的TICK流
        self.kline_key = f"kline:{symbol}:1m"  # 历史K线存储
        self.current_kline_channel = f"current_kline:{symbol}:m1"  # 当前K线Pub/Sub频道
        
        # K线构建器（支持多时间周期）
        active_timeframes = ['M1']  # 默认只激活M1
        self.kline_builder = KlineBuilderEnhanced(
            symbol=self.symbol,
            active_timeframes=active_timeframes
        )
        
        # 状态
        self.last_id = b'$'  # 从最新消息开始读取（使用字节格式）
        self.stats = {
            'ticks_processed': 0,
            'klines_closed': 0,
            'klines_pushed': 0,
        }
        
        # 统计日志相关
        self.last_stats_time = time.time()
        self.last_kline_info = None  # 记录最近推送的K线信息
        self.stats_interval = 10  # 统计日志输出间隔（秒）
        
        logger.info(f"K线服务初始化成功: {symbol}")
        logger.info(f"  - 输入: {self.validated_tick_stream_key}")
        logger.info(f"  - 历史K线: {self.kline_key}")
        logger.info(f"  - 当前K线: {self.current_kline_channel}")
    
    def _load_historical_klines(self):
        """从Redis加载历史K线数据，初始化K线构建器"""
        try:
            # 从Redis读取最近的历史K线
            klines_json = self.r_text.zrange(self.kline_key, -2880, -1)  # 最近2880根（2天）
            
            if klines_json:
                historical_klines = []
                for kline_json in klines_json:
                    try:
                        kline = json.loads(kline_json)
                        historical_klines.append(kline)
                    except json.JSONDecodeError:
                        continue
                
                if historical_klines:
                    # 按时间排序
                    historical_klines.sort(key=lambda x: x.get('time', 0))
                    
                    # 加载到K线构建器
                    self.kline_builder.load_history('M1', historical_klines)
                    logger.info(f"✅ 已加载 {len(historical_klines)} 根历史K线")
                else:
                    logger.warning("⚠️ Redis中的K线数据格式无效")
            else:
                logger.info("ℹ️ Redis中没有历史K线数据，将从TICK流开始构建")
        
        except Exception as e:
            logger.warning(f"加载历史K线失败（非关键）: {e}")
    
    def _save_closed_kline_to_redis(self, closed_kline, timeframe: str = 'M1'):
        """
        存储闭合的K线到Redis
        
        Args:
            closed_kline: 闭合的K线（可能是字典或numpy array）
            timeframe: 时间周期（'M1', 'M5', 'H1'等）
        """
        try:
            # 🔴 修复：处理两种格式（字典或numpy array）
            import numpy as np
            
            if isinstance(closed_kline, np.ndarray):
                # 如果是numpy array（structured array），转换为字典
                if len(closed_kline) > 0:
                    kline_elem = closed_kline[0]
                    # 检查是否有volume字段
                    volume = 0
                    if 'volume' in closed_kline.dtype.names:
                        volume = int(kline_elem['volume'])
                    
                    kline_dict = {
                        'time': int(kline_elem['time']),
                        'open': float(kline_elem['open']),
                        'high': float(kline_elem['high']),
                        'low': float(kline_elem['low']),
                        'close': float(kline_elem['close']),
                        'volume': volume,
                        'real_volume': 0,
                        'is_closed': True
                    }
                else:
                    logger.warning(f"K线服务: 收到空的numpy array，跳过存储")
                    return
            elif isinstance(closed_kline, dict):
                # 如果是字典，直接使用
                kline_dict = {
                    'time': int(closed_kline.get('time', 0)),
                    'open': float(closed_kline.get('open', 0)),
                    'high': float(closed_kline.get('high', 0)),
                    'low': float(closed_kline.get('low', 0)),
                    'close': float(closed_kline.get('close', 0)),
                    'volume': int(closed_kline.get('volume', 0)),
                    'real_volume': int(closed_kline.get('real_volume', 0)),
                    'is_closed': True
                }
            else:
                logger.warning(f"K线服务: 未知的K线数据格式: {type(closed_kline)}")
                return
            
            # 如果K线数据无效，跳过
            if kline_dict['time'] == 0 or kline_dict['close'] == 0:
                return
            
            kline_key = f"kline:{self.symbol}:{timeframe.lower()}"
            kline_json = json.dumps(kline_dict, ensure_ascii=False)
            
            # 去重：先删除相同时间戳的旧数据
            kline_time = kline_dict['time']
            self.r_text.zremrangebyscore(kline_key, kline_time, kline_time)
            
            # 使用ZADD存储新数据（确保时间戳唯一）
            self.r_text.zadd(kline_key, {kline_json: kline_time})
            
            # 保留最近2880根（2天M1数据）
            current_count = self.r_text.zcard(kline_key)
            if current_count > 2880:
                remove_count = current_count - 2880
                self.r_text.zremrangebyrank(kline_key, 0, remove_count - 1)
            
            # 发布Pub/Sub通知（供API Server订阅）
            try:
                self.r_text.publish(
                    f"kline:{self.symbol}:{timeframe.lower()}",
                    kline_json
                )
            except Exception as e:
                logger.debug(f"K线Pub/Sub通知失败（非关键）: {e}")
            
            self.stats['klines_closed'] += 1
            logger.debug(f"✅ K线已存储: {timeframe} @ {kline_dict['time']} (C:{kline_dict['close']:.2f})")
        
        except Exception as e:
            logger.error(f"存储K线到Redis失败: {e}")
    
    def _push_current_kline_to_redis(self, current_kline: Dict[str, Any], timeframe: str = 'M1'):
        """
        实时推送当前未闭合的K线到Redis Pub/Sub
        
        Args:
            current_kline: 当前未闭合的K线字典
            timeframe: 时间周期（'M1', 'M5', 'H1'等）
        """
        try:
            # 转换为标准格式
            kline_dict = {
                'time': int(current_kline.get('time', 0)),
                'open': float(current_kline.get('open', 0)),
                'high': float(current_kline.get('high', 0)),
                'low': float(current_kline.get('low', 0)),
                'close': float(current_kline.get('close', 0)),
                'volume': int(current_kline.get('volume', 0)),
                'real_volume': 0,
                'is_closed': False  # 标记为未闭合
            }
            
            # 如果K线数据无效，跳过
            if kline_dict['time'] == 0 or kline_dict['close'] == 0:
                return
            
            # 推送到Redis Pub/Sub频道（实时跳动）
            current_kline_channel = f"current_kline:{self.symbol}:{timeframe.lower()}"
            kline_json = json.dumps(kline_dict, ensure_ascii=False)
            self.r_text.publish(current_kline_channel, kline_json)
            
            # 同时更新Redis中的当前K线快照（供API查询）
            current_kline_key = f"current_kline:{self.symbol}:{timeframe.lower()}:snapshot"
            self.r_text.set(current_kline_key, kline_json, ex=120)  # 2分钟过期
            
            self.stats['klines_pushed'] += 1
            
            # 记录最近推送的K线信息（用于统计日志）
            self.last_kline_info = {
                'time': kline_dict['time'],
                'close': kline_dict['close'],
                'high': kline_dict['high'],
                'low': kline_dict['low'],
                'open': kline_dict['open']
            }
            
            # 🔴 修复：定期输出统计日志，避免每次推送都记录（减少日志刷屏）
            current_time = time.time()
            if current_time - self.last_stats_time >= self.stats_interval:
                self._log_stats()
                self.last_stats_time = current_time
        
        except Exception as e:
            logger.error(f"推送当前K线失败: {e}")
    
    def _log_stats(self):
        """输出统计日志（定期调用，避免日志刷屏）"""
        try:
            if self.last_kline_info:
                logger.info(
                    f"📊 K线服务统计 [{self.stats_interval}秒] - "
                    f"处理TICK: {self.stats['ticks_processed']}, "
                    f"闭合K线: {self.stats['klines_closed']}, "
                    f"推送K线: {self.stats['klines_pushed']}, "
                    f"当前K线: time={self.last_kline_info['time']}, "
                    f"O={self.last_kline_info['open']:.2f}, "
                    f"H={self.last_kline_info['high']:.2f}, "
                    f"L={self.last_kline_info['low']:.2f}, "
                    f"C={self.last_kline_info['close']:.2f}"
                )
            else:
                logger.info(
                    f"📊 K线服务统计 [{self.stats_interval}秒] - "
                    f"处理TICK: {self.stats['ticks_processed']}, "
                    f"闭合K线: {self.stats['klines_closed']}, "
                    f"推送K线: {self.stats['klines_pushed']}"
                )
        except Exception as e:
            logger.debug(f"输出统计日志失败: {e}")
    
    def _process_tick(self, tick_data: Dict[str, Any]):
        """
        处理单个TICK数据，构建K线
        
        Args:
            tick_data: TICK数据字典
        """
        try:
            # 解析TICK数据
            time_msc = tick_data.get('time_msc', 0)
            if time_msc == 0:
                return
            
            # 获取价格
            last_price = float(tick_data.get('last', 0.0))
            bid_price = float(tick_data.get('bid', 0.0))
            ask_price = float(tick_data.get('ask', 0.0))
            
            if last_price > 0:
                price = last_price
            elif bid_price > 0 and ask_price > 0:
                price = (bid_price + ask_price) / 2.0
            elif bid_price > 0:
                price = bid_price
            elif ask_price > 0:
                price = ask_price
            else:
                logger.warning(f"K线服务: TICK价格无效，跳过 (Seq: {tick_data.get('seq')})")
                return
            
            volume = float(tick_data.get('volume', 0.0))
            
            # 构建TICK字典
            tick_dict = {
                'time_msc': time_msc,
                'time': int(time_msc / 1000),
                'last': price,
                'volume': int(volume),
                'bid': tick_data.get('bid', price),
                'ask': tick_data.get('ask', price)
            }
            
            # 处理TICK，构建K线
            closed_klines = self.kline_builder.process_tick(tick_dict)
            
            # 如果M1周期K线收盘，存储到Redis
            closed_kline_m1 = closed_klines.get('M1')
            if closed_kline_m1 is not None:
                self._save_closed_kline_to_redis(closed_kline_m1, 'M1')
            
            # 实时推送当前未闭合的K线（每次TICK都更新）
            current_kline_m1 = self.kline_builder.get_current_candle('M1')
            if current_kline_m1 and current_kline_m1.get('time', 0) > 0:
                self._push_current_kline_to_redis(current_kline_m1, 'M1')
            
            self.stats['ticks_processed'] += 1
        
        except Exception as e:
            logger.error(f"K线服务: 处理TICK数据错误: {e}")
    
    def _data_receiver_loop(self):
        """数据接收循环：从Redis Stream读取TICK数据"""
        logger.info("K线服务: 数据接收线程已启动")
        
        # 🔴 修复：确保使用字节格式的stream key
        validated_tick_stream_key_bytes = self.validated_tick_stream_key.encode('utf-8') if isinstance(self.validated_tick_stream_key, str) else self.validated_tick_stream_key
        
        while not self.stop_event.is_set():
            try:
                # 从Redis Stream读取TICK数据
                # 🔴 修复：确保last_id是字节格式
                last_id_bytes = self.last_id.encode('utf-8') if isinstance(self.last_id, str) else self.last_id
                messages = self.redis_client.xread(
                    {validated_tick_stream_key_bytes: last_id_bytes},
                    count=100,  # 批量读取
                    block=1000  # 阻塞1秒
                )
                
                if messages:
                    for stream_name, stream_messages in messages:
                        for msg_id, msg_data in stream_messages:
                            try:
                                # 解析TICK数据
                                if b'value' in msg_data:
                                    tick_json = msg_data[b'value'].decode('utf-8')
                                else:
                                    # 兼容文本模式
                                    tick_json = msg_data.get('value', '{}')
                                
                                tick_data = json.loads(tick_json)
                                
                                # 处理TICK
                                self._process_tick(tick_data)
                                
                                # 更新last_id
                                self.last_id = msg_id
                            
                            except json.JSONDecodeError as e:
                                logger.warning(f"K线服务: TICK数据JSON解析失败: {e}")
                            except Exception as e:
                                logger.error(f"K线服务: 处理消息失败: {e}")
            
            except redis.exceptions.ConnectionError as e:
                logger.error(f"K线服务: Redis连接错误: {e}")
                # 🔴 修复：尝试重新创建Redis连接
                try:
                    self.redis_client = redis.Redis(
                        host=REDIS_CONFIG.get('host', 'localhost'),
                        port=REDIS_CONFIG.get('port', 6379),
                        db=REDIS_CONFIG.get('db', 0),
                        decode_responses=False
                    )
                    self.redis_client.ping()
                    logger.info("K线服务: Redis连接已恢复")
                except Exception as reconnect_error:
                    logger.warning(f"K线服务: Redis重连失败: {reconnect_error}")
                time.sleep(5)  # 等待5秒后重试
            except redis.exceptions.TimeoutError as e:
                logger.warning(f"K线服务: Redis超时: {e}")
                time.sleep(2)  # 超时后短暂等待
            except Exception as e:
                logger.error(f"K线服务: 数据接收循环错误: {e}")
                time.sleep(1)
        
        logger.info("K线服务: 数据接收线程已停止")
    
    def start(self):
        """启动K线服务"""
        logger.info("=" * 70)
        logger.info("启动K线服务")
        logger.info("=" * 70)
        
        # 加载历史K线数据
        self._load_historical_klines()
        
        # 启动数据接收线程
        self.data_receiver_thread = threading.Thread(
            target=self._data_receiver_loop,
            daemon=True,
            name="KlineService-DataReceiver"
        )
        self.data_receiver_thread.start()
        
        logger.info("✅ K线服务已启动")
        logger.info(f"  - 已处理TICK: {self.stats['ticks_processed']}")
        logger.info(f"  - 已闭合K线: {self.stats['klines_closed']}")
        logger.info(f"  - 已推送K线: {self.stats['klines_pushed']}")
    
    def stop(self):
        """停止K线服务"""
        logger.info("正在停止K线服务...")
        self.stop_event.set()
        
        # 等待线程结束
        if hasattr(self, 'data_receiver_thread'):
            self.data_receiver_thread.join(timeout=5)
        
        logger.info("✅ K线服务已停止")
        logger.info(f"  - 总计处理TICK: {self.stats['ticks_processed']}")
        logger.info(f"  - 总计闭合K线: {self.stats['klines_closed']}")
        logger.info(f"  - 总计推送K线: {self.stats['klines_pushed']}")


def main():
    """主函数：独立运行K线服务"""
    import signal
    
    kline_service = KlineService(symbol="BTCUSDm")
    
    def signal_handler(sig, frame):
        logger.info("收到停止信号，正在优雅关闭...")
        kline_service.stop()
        exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    kline_service.start()
    
    # 保持运行
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        signal_handler(None, None)


if __name__ == "__main__":
    main()

