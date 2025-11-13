#date: 2025-11-13T17:03:12Z
#url: https://api.github.com/gists/d68fdd03a1f1096971a5f4ed71530eb2
#owner: https://api.github.com/users/wangwei334455

"""
Redis数据处理器
负责数据的存储和读取

【设计说明】
- 用途：历史数据持久化和查询（Sorted Set）
- decode_responses=True：自动将Redis返回的字节串解码为Python字符串
- JSON序列化：使用ensure_ascii=False，支持UTF-8编码（包含中文等非ASCII字符）
- 与RealtimeCollector分工：
  * RedisHandler：负责Sorted Set历史数据持久化（decode_responses=True）
  * RealtimeCollector：负责Stream实时传输（使用独立的Redis客户端，decode_responses=False）
"""
import redis
import json
from typing import Dict, List, Optional
from loguru import logger


class RedisHandler:
    """
    Redis数据处理器
    
    【职责分工】
    1. Sorted Set (kline:BTCUSDm:1m, tick:BTCUSDm): 历史数据持久化
    2. String (tick:BTCUSDm:latest): 最新数据快照（O(1)查询）
    3. 滚动删除：自动维护固定大小的数据窗口（K线2880根，TICK 3000条）
    
    【注意】
    - 本Handler使用decode_responses=True，适合历史数据查询
    - RealtimeCollector的Stream写入使用独立的Redis客户端（decode_responses=False）
    """
    
    def __init__(self, config: Dict, keys: Dict):
        """
        初始化Redis处理器
        
        Args:
            config: Redis配置
            keys: Redis键配置
        """
        self.config = config
        self.keys = keys
        self.client = None
        self.connected = False
    
    def connect(self) -> bool:
        """
        连接Redis
        
        【配置说明】
        - decode_responses=True: 自动解码为Python字符串，适合JSON数据查询
        - 与RealtimeCollector的Stream客户端分离，避免编码冲突
        """
        try:
            self.client = redis.Redis(
                host=self.config['host'],
                port=self.config['port'],
                db=self.config['db'],
                password= "**********"
                decode_responses=True  # 自动解码为Python字符串，适合JSON查询
            )
            
            # 测试连接
            self.client.ping()
            self.connected = True
            logger.info(f"Redis连接成功: {self.config['host']}:{self.config['port']}")
            return True
            
        except Exception as e:
            logger.error(f"Redis连接失败: {str(e)}")
            self.connected = False
            return False
    
    def disconnect(self):
        """断开Redis连接"""
        if self.client:
            self.client.close()
            self.connected = False
            logger.info("Redis连接已断开")
    
    def save_klines(self, klines: List[Dict], max_buffer: int = 2880) -> bool:
        """
        保存K线数据到Redis (批量)
        
        Args:
            klines: K线数据列表
            max_buffer: 最大缓存数量（默认2880，即2天M1 K线：2天×1440分钟=2880根）
            
        Returns:
            bool: 是否成功
        """
        if not self.connected:
            logger.error("未连接到Redis")
            return False
        
        try:
            key = self.keys['kline_data']
            
            # 使用pipeline批量操作
            pipe = self.client.pipeline()
            
            for kline in klines:
                # 使用时间戳作为score
                score = kline['time']
                # 【JSON序列化】ensure_ascii=False确保UTF-8编码，与decode_responses=True配合良好
                value = json.dumps(kline, ensure_ascii=False)
                pipe.zadd(key, {value: score})
            
            # 执行批量操作
            pipe.execute()
            
            # 【滚动删除】保持固定大小（保留最新的数据）
            # 使用ZREMRANGEBYRANK删除最旧的数据，实现FIFO滚动窗口
            current_count = self.client.zcard(key)
            if current_count > max_buffer:
                remove_count = current_count - max_buffer
                self.client.zremrangebyrank(key, 0, remove_count - 1)
                logger.debug(f"K线数据滚动删除: 删除 {remove_count} 条旧数据，保留 {max_buffer} 条")
            
            logger.debug(f"成功保存 {len(klines)} 条K线数据到Redis")
            return True
            
        except Exception as e:
            logger.error(f"保存K线数据到Redis失败: {str(e)}")
            return False
    
    def save_kline(self, kline: Dict, max_buffer: int = 2880) -> bool:
        """
        保存单条K线数据到Redis
        
        Args:
            kline: K线数据
            max_buffer: 最大缓存数量（默认2880，即2天M1 K线）
            
        Returns:
            bool: 是否成功
        """
        return self.save_klines([kline], max_buffer)
    
    def save_tick(self, tick: Dict, max_buffer: int = 10000, max_days: int = 30) -> bool:
        """
        保存Tick数据到Redis (使用有序集合)
        
        按照官方最佳实践：
        1. 使用sorted set存储，score为时间戳（毫秒）
        2. 定期清理超过30天的旧数据（由cleanup_old_ticks方法处理）
        3. 备用数量限制防止内存溢出（不再在每次保存时执行）
        
        Args:
            tick: Tick数据
            max_buffer: 最大缓存数量（备用限制，不再在每次保存时强制执行）
            max_days: 最大保存天数（仅用于cleanup_old_ticks，此处不直接使用）
            
        Returns:
            bool: 是否成功
        """
        if not self.connected:
            logger.error("未连接到Redis")
            return False
        
        try:
            key = self.keys['tick_data']
            
            # 使用时间戳（毫秒）作为score
            score = tick.get('time_msc', tick['time'] * 1000)
            # 【JSON序列化】ensure_ascii=False确保UTF-8编码
            value = json.dumps(tick, ensure_ascii=False)
            
            # 添加到有序集合（历史存储）
            self.client.zadd(key, {value: score})
            
            # 同时更新最新tick（String快照，O(1)查询）
            self.client.set(self.keys['latest_tick'], value)
            
            # 注意：Stream写入由realtime_collector实时处理，这里只负责持久化到Sorted Set
            
            return True
            
        except Exception as e:
            logger.error(f"保存Tick数据到Redis失败: {str(e)}")
            return False
    
    def save_ticks_batch(self, ticks: List[Dict], max_buffer: int = 3000, symbol: Optional[str] = None) -> int:
        """
        批量保存Tick数据到Redis (使用Pipeline实现三向高性能写入)
        
        【三向写入】
        1. Sorted Set (ZADD): 历史持久化，支持范围查询
        2. String (SET): 最新TICK快照，O(1)查询
        3. Stream (XADD): 实时数据流，供L2策略层消费
        
        Args:
            ticks: Tick数据列表（已包含seq和checksum）
            max_buffer: Sorted Set最大缓存数量（默认3000）
            symbol: 交易品种（可选，用于构建Stream key）
            
        Returns:
            int: 成功写入的TICK数量
        """
        if not self.connected:
            logger.error("未连接到Redis")
            return 0
        
        if not ticks:
            return 0
        
        try:
            # 键名定义
            tick_data_key = self.keys['tick_data']  # Sorted Set (历史/查询)
            latest_tick_key = self.keys['latest_tick']  # String (最新快照)
            
            # Stream key（如果提供了symbol，使用动态key；否则使用默认key）
            if symbol:
                tick_stream_key = f"tick:{symbol}:stream"
            else:
                # 从tick_data_key推断symbol（例如：tick:BTCUSDm -> tick:BTCUSDm:stream）
                tick_stream_key = self.keys.get('tick_stream', tick_data_key.replace(':data', ':stream'))
            
            latest_tick = ticks[-1]  # 最新的一条TICK
            
            # 使用Pipeline批量操作
            pipe = self.client.pipeline()
            
            # 1. ZADD 批量写入 Sorted Set（历史持久化）
            for tick in ticks:
                score = tick.get('time_msc', tick.get('time', 0) * 1000)
                value = json.dumps(tick, ensure_ascii=False)
                pipe.zadd(tick_data_key, {value: score})
            
            # 2. SET 最新TICK快照（O(1)查询）
            latest_value = json.dumps(latest_tick, ensure_ascii=False)
            pipe.set(latest_tick_key, latest_value)
            
            # 3. XADD 批量写入Stream（实时分发给L2策略层）
            # Stream的每个entry存储一个TICK的完整JSON字符串作为field value
            for tick in ticks:
                tick_json = json.dumps(tick, ensure_ascii=False)
                # XADD的value必须是字典，这里将整个TICK JSON作为'value'字段
                pipe.xadd(
                    tick_stream_key,
                    {'value': tick_json},
                    id='*',  # 自动生成ID
                    maxlen=1000,  # 保持Stream长度，防止无限增长
                    approximate=True  # 允许近似MAXLEN（性能优化）
                )
            
            # 4. Sorted Set滚动删除（保留最新的N条）
            current_count = self.client.zcard(tick_data_key)
            if current_count > max_buffer:
                remove_count = current_count - max_buffer
                pipe.zremrangebyrank(tick_data_key, 0, remove_count - 1)
            
            # 执行Pipeline（一次网络往返完成所有操作）
            pipe.execute()
            
            logger.debug(f"✓ 批量写入 {len(ticks)} 条TICK到Redis（Sorted Set + Stream + String）")
            
            return len(ticks)
            
        except Exception as e:
            logger.error(f"批量保存Tick数据到Redis失败: {str(e)}")
            return 0
    
    def cleanup_old_ticks(self, max_days: int = 30) -> int:
        """
        批量清理超过指定天数的旧TICK数据
        
        按照官方最佳实践：定期批量删除，而不是每次保存都删除
        
        Args:
            max_days: 最大保存天数
            
        Returns:
            int: 删除的TICK数量
        """
        if not self.connected:
            logger.error("未连接到Redis")
            return 0
        
        try:
            key = self.keys['tick_data']
            
            # 计算截止时间（毫秒）
            import time
            current_time_msc = int(time.time() * 1000)
            cutoff_time_msc = current_time_msc - (max_days * 24 * 60 * 60 * 1000)
            
            # 批量删除超过30天的旧TICK
            deleted_count = self.client.zremrangebyscore(key, '-inf', cutoff_time_msc)
            
            if deleted_count > 0:
                logger.info(f"清理完成: 删除了 {deleted_count} 个超过{max_days}天的旧TICK")
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"清理旧TICK数据失败: {str(e)}")
            return 0
    
    def get_klines(self, start_index: int = 0, count: int = -1) -> List[Dict]:
        """
        从Redis获取K线数据
        
        Args:
            start_index: 起始索引（0表示最旧的）
            count: 获取数量（-1表示全部）
            
        Returns:
            List[Dict]: K线数据列表
        """
        if not self.connected:
            logger.error("未连接到Redis")
            return []
        
        try:
            key = self.keys['kline_data']
            
            # 从有序集合获取数据
            if count == -1:
                raw_data = self.client.zrange(key, start_index, -1)
            else:
                end_index = start_index + count - 1
                raw_data = self.client.zrange(key, start_index, end_index)
            
            # 解析JSON数据
            klines = [json.loads(item) for item in raw_data]
            
            logger.debug(f"成功获取 {len(klines)} 条K线数据")
            return klines
            
        except Exception as e:
            logger.error(f"获取K线数据失败: {str(e)}")
            return []
    
    def get_latest_klines(self, count: int = 100) -> List[Dict]:
        """
        获取最新的N条K线数据
        
        Args:
            count: 获取数量
            
        Returns:
            List[Dict]: K线数据列表
        """
        if not self.connected:
            logger.error("未连接到Redis")
            return []
        
        try:
            key = self.keys['kline_data']
            
            # 从有序集合获取最新的N条数据
            raw_data = self.client.zrange(key, -count, -1)
            
            # 解析JSON数据
            klines = [json.loads(item) for item in raw_data]
            
            logger.debug(f"成功获取最新 {len(klines)} 条K线数据")
            return klines
            
        except Exception as e:
            logger.error(f"获取最新K线数据失败: {str(e)}")
            return []
    
    def get_klines_count(self) -> int:
        """获取K线数据总数"""
        if not self.connected:
            return 0
        
        try:
            key = self.keys['kline_data']
            return self.client.zcard(key)
        except Exception as e:
            logger.error(f"获取K线数量失败: {str(e)}")
            return 0
    
    def get_ticks(self, start_index: int = 0, count: int = -1) -> List[Dict]:
        """
        从Redis获取Tick数据
        
        Args:
            start_index: 起始索引
            count: 获取数量（-1表示全部）
            
        Returns:
            List[Dict]: Tick数据列表
        """
        if not self.connected:
            logger.error("未连接到Redis")
            return []
        
        try:
            key = self.keys['tick_data']
            
            # 从有序集合获取数据
            if count == -1:
                raw_data = self.client.zrange(key, start_index, -1)
            else:
                end_index = start_index + count - 1
                raw_data = self.client.zrange(key, start_index, end_index)
            
            # 解析JSON数据
            ticks = [json.loads(item) for item in raw_data]
            
            logger.debug(f"成功获取 {len(ticks)} 条Tick数据")
            return ticks
            
        except Exception as e:
            logger.error(f"获取Tick数据失败: {str(e)}")
            return []
    
    def get_latest_tick(self) -> Optional[Dict]:
        """
        获取最新的Tick数据
        
        Returns:
            Optional[Dict]: Tick数据，如果不存在则返回None
        """
        if not self.connected:
            logger.error("未连接到Redis")
            return None
        
        try:
            key = self.keys['latest_tick']
            tick_json = self.client.get(key)
            
            if tick_json:
                return json.loads(tick_json)
            return None
            
        except Exception as e:
            logger.error(f"获取最新Tick数据失败: {str(e)}")
            return None
    
    def get_ticks_count(self) -> int:
        """获取Tick数据总数"""
        if not self.connected:
            return 0
        
        try:
            key = self.keys['tick_data']
            return self.client.zcard(key)
        except Exception as e:
            logger.error(f"获取Tick数量失败: {str(e)}")
            return 0
    
    def save_collector_status(self, status: Dict) -> bool:
        """
        保存采集器状态
        
        Args:
            status: 状态信息
            
        Returns:
            bool: 是否成功
        """
        if not self.connected:
            logger.error("未连接到Redis")
            return False
        
        try:
            key = self.keys['collector_status']
            value = json.dumps(status, ensure_ascii=False)
            self.client.set(key, value)
            
            logger.debug("成功保存采集器状态")
            return True
            
        except Exception as e:
            logger.error(f"保存采集器状态失败: {str(e)}")
            return False
    
    def get_collector_status(self) -> Optional[Dict]:
        """
        获取采集器状态
        
        Returns:
            Optional[Dict]: 状态信息，如果不存在则返回None
        """
        if not self.connected:
            logger.error("未连接到Redis")
            return None
        
        try:
            key = self.keys['collector_status']
            status_json = self.client.get(key)
            
            if status_json:
                return json.loads(status_json)
            return None
            
        except Exception as e:
            logger.error(f"获取采集器状态失败: {str(e)}")
            return None    return None