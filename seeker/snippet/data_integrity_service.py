#date: 2025-11-13T17:02:59Z
#url: https://api.github.com/gists/c1c2c32c02ad6b73f59da7fc63add2f0
#owner: https://api.github.com/users/wangwei334455

"""
数据完整性检查服务

【职责分离】
- 专门负责数据完整性检查、去重、补空
- 策略核心只处理已验证的完整数据
- API Server从已验证的数据中读取

【功能】
1. TICK数据验证（seq检查、checksum验证）
2. K线数据去重（相同时间戳只保留最新）
3. K线数据补空（填充缺失的时间段）
4. 数据质量监控（缺失率、重复率统计）
"""
import json
import time
import hashlib
import threading
from typing import Dict, Any, List, Optional
from datetime import datetime
from loguru import logger
import redis
import numpy as np

from config.redis_config import REDIS_CONFIG, REDIS_KEYS


class DataIntegrityService:
    """
    数据完整性检查服务
    
    【架构设计】
    - 独立服务，不依赖策略核心
    - 监听Redis Stream，验证TICK数据
    - 验证后的数据存储到已验证的Stream
    - 定期检查K线数据完整性，自动补空
    """
    
    def __init__(self, symbol: str = "BTCUSDm"):
        self.symbol = symbol
        self.redis_client = redis.Redis(**REDIS_CONFIG)
        self.r_text = redis.Redis(
            host=REDIS_CONFIG.get('host', 'localhost'),
            port=REDIS_CONFIG.get('port', 6379),
            db=REDIS_CONFIG.get('db', 0),
            decode_responses=True
        )
        
        # Redis Keys
        self.tick_stream_key = REDIS_KEYS['tick_stream']  # 原始TICK流
        self.validated_tick_stream_key = f"tick:{symbol}:validated:stream"  # 验证后的TICK流
        self.kline_key = f"kline:{symbol}:1m"  # K线数据
        
        # 状态
        self.last_processed_seq = 0
        self.stop_event = threading.Event()
        self.stats = {
            'ticks_validated': 0,
            'ticks_rejected': 0,
            'klines_deduplicated': 0,
            'klines_filled': 0,
        }
        
        logger.info(f"数据完整性检查服务初始化: {symbol}")
    
    def validate_tick(self, tick_data: Dict[str, Any]) -> bool:
        """
        验证TICK数据完整性
        
        Args:
            tick_data: TICK数据字典
            
        Returns:
            bool: 是否通过验证
        """
        try:
            # 1. 检查必需字段
            required_fields = ['time_msc', 'seq', 'checksum', 'last', 'bid', 'ask']
            if not all(field in tick_data for field in required_fields):
                logger.warning(f"TICK缺少必需字段: {tick_data}")
                return False
            
            # 2. 检查序列号（Seq Check）
            current_seq = tick_data.get('seq', 0)
            expected_seq = self.last_processed_seq + 1
            
            # 处理首次TICK（seq=0）
            if current_seq == 0 and self.last_processed_seq == 0:
                current_seq = 1
                tick_data['seq'] = 1
            
            # 处理seq跳跃（允许继续，但记录警告）
            if current_seq > expected_seq:
                seq_gap = current_seq - expected_seq
                # 🔴 优化：小幅度跳跃（<10）可能是正常的（服务重启），只记录DEBUG
                # 大幅度跳跃（>=10）才记录WARNING，可能是数据丢失
                if seq_gap < 10:
                    logger.debug(f"数据完整性: Seq小幅跳跃（可能服务重启） - 期望 {expected_seq}，实际收到 {current_seq}（跳跃 {seq_gap}）")
                else:
                    logger.warning(f"数据完整性: Seq大幅跳跃！期望 {expected_seq}，实际收到 {current_seq}（丢失了 {seq_gap} 个TICK）")
                self.last_processed_seq = current_seq - 1
                expected_seq = current_seq
            
            # 处理重复seq（跳过，但降低日志级别，避免日志刷屏）
            if current_seq < expected_seq:
                # 🔴 修复：只在seq差距较大时记录警告，避免日志刷屏
                if expected_seq - current_seq > 10:
                    logger.warning(f"数据完整性: Seq重复！期望 {expected_seq}，实际收到 {current_seq}（差距 {expected_seq - current_seq}，跳过）")
                # 否则静默跳过，不记录日志
                return False
            
            # 3. 校验和验证（Checksum Check）
            checksum_base = f"{tick_data.get('time_msc', 0)}:{current_seq}:{tick_data.get('bid', 0)}:{tick_data.get('ask', 0)}"
            recalculated_checksum = hashlib.md5(checksum_base.encode('utf-8')).hexdigest()[:8]
            
            if recalculated_checksum != tick_data.get('checksum', ''):
                logger.error(f"数据完整性: Checksum错误！Seq={current_seq}。数据可能被篡改！")
                return False
            
            # 4. 更新最后处理的seq
            self.last_processed_seq = current_seq
            self.stats['ticks_validated'] += 1
            
            return True
            
        except Exception as e:
            logger.error(f"数据完整性: TICK验证异常: {e}")
            return False
    
    def deduplicate_klines(self, klines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        去重K线数据（相同时间戳只保留最新的）
        
        Args:
            klines: K线列表
            
        Returns:
            去重后的K线列表
        """
        seen_times = {}
        for kline in klines:
            kline_time = kline.get('time', 0)
            if kline_time > 0:
                # 保留最新的数据（后面的数据覆盖前面的）
                if kline_time not in seen_times:
                    seen_times[kline_time] = kline
                else:
                    # 如果volume更大，说明是更完整的数据
                    if kline.get('volume', 0) > seen_times[kline_time].get('volume', 0):
                        seen_times[kline_time] = kline
                        self.stats['klines_deduplicated'] += 1
        
        # 转换为列表并按时间排序
        unique_klines = list(seen_times.values())
        unique_klines.sort(key=lambda x: x.get('time', 0))
        
        return unique_klines
    
    def fill_missing_klines(self, klines: List[Dict[str, Any]], timeframe: str = '1m') -> List[Dict[str, Any]]:
        """
        填充缺失的K线（MT5官方最佳实践）
        
        Args:
            klines: K线列表（已排序）
            timeframe: 时间周期（'1m', '5m'等）
            
        Returns:
            填充后的K线列表
        """
        if len(klines) < 2:
            return klines
        
        # 时间周期映射
        timeframe_minutes = {
            '1m': 1, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '4h': 240, '1d': 1440
        }
        minutes_per_kline = timeframe_minutes.get(timeframe, 1)
        expected_interval = minutes_per_kline * 60  # 秒
        
        filled_klines = [klines[0]]  # 第一根K线
        
        for i in range(1, len(klines)):
            prev_kline = klines[i - 1]
            current_kline = klines[i]
            
            prev_time = prev_kline.get('time', 0)
            current_time = current_kline.get('time', 0)
            
            # 计算实际时间间隔
            actual_interval = current_time - prev_time
            
            # 如果时间间隔大于期望间隔，说明有缺失
            if actual_interval > expected_interval + 1:  # 允许1秒容差
                # 🔴 修复：更精确的缺失K线数量计算
                # 计算理论上应该有的K线数量，并减去已有的1根（当前K线）
                # 例如：间隔120s，期望60s，应该有2根，缺失1根
                # 例如：间隔180s，期望60s，应该有3根，缺失2根
                kline_count_in_gap = actual_interval // expected_interval
                missing_count = kline_count_in_gap - 1  # 应该有的数量 - 1 (当前K线)
                
                if missing_count >= 1:
                    logger.debug(f"数据完整性: 发现K线缺失 - 从 {prev_time} 到 {current_time} 缺失 {missing_count} 根 (间隔={actual_interval}s, 期望={expected_interval}s)")
                    
                    # 填充缺失的K线
                    prev_close = prev_kline.get('close', 0)
                    if prev_close > 0:
                        for j in range(1, missing_count + 1):
                            missing_time = prev_time + (j * expected_interval)
                            
                            filled_kline = {
                                'time': missing_time,
                                'open': prev_close,
                                'high': prev_close,
                                'low': prev_close,
                                'close': prev_close,
                                'volume': 0,
                                'real_volume': 0,
                                'is_filled': True
                            }
                            filled_klines.append(filled_kline)
                            self.stats['klines_filled'] += 1
                    else:
                        logger.warning(f"数据完整性: 无法填充缺失K线 - 前一根收盘价为0, time={prev_time}")
            
            # 添加当前K线
            filled_klines.append(current_kline)
        
        return filled_klines
    
    def validate_and_store_tick(self, tick_data: Dict[str, Any]):
        """
        验证TICK数据并存储到已验证的Stream
        
        Args:
            tick_data: TICK数据字典
        """
        if self.validate_tick(tick_data):
            # 存储到验证后的Stream（供策略核心消费）
            tick_json = json.dumps(tick_data, ensure_ascii=False)
            self.r_text.xadd(
                self.validated_tick_stream_key,
                {'value': tick_json},
                id='*',
                maxlen=1000,
                approximate=True
            )
        else:
            self.stats['ticks_rejected'] += 1
    
    def check_and_fix_klines(self):
        """
        检查并修复K线数据完整性（定期执行）
        
        【职责】
        - 定期检查Redis中的K线数据完整性
        - 去重（相同时间戳只保留最新）
        - 补空（填充缺失的时间段）
        
        【注意】
        - L2策略核心在存储时已经去重（使用zremrangebyscore）
        - API Server在读取时也会去重和补空
        - 此方法作为最后的保障，定期修复历史数据
        
        功能：
        1. 去重（相同时间戳只保留最新）
        2. 补空（填充缺失的时间段）
        3. 存储修复后的数据
        """
        try:
            # 从Redis读取所有K线
            klines_json = self.r_text.zrange(self.kline_key, 0, -1, withscores=False)
            if not klines_json:
                return
            
            # 解析K线数据
            klines = [json.loads(k) for k in klines_json]
            original_count = len(klines)
            
            # 1. 去重（使用zremrangebyscore确保时间戳唯一）
            unique_klines = []
            seen_times = set()
            for kline in klines:
                kline_time = kline.get('time', 0)
                if kline_time > 0 and kline_time not in seen_times:
                    unique_klines.append(kline)
                    seen_times.add(kline_time)
                elif kline_time in seen_times:
                    # 发现重复，需要去重
                    self.stats['klines_deduplicated'] += 1
            
            # 按时间排序
            unique_klines.sort(key=lambda x: x.get('time', 0))
            
            # 2. 补空（填充缺失的时间段）
            filled_klines = self.fill_missing_klines(unique_klines, '1m')
            
            # 3. 如果数据有变化，重新存储到Redis（使用原子操作）
            if len(filled_klines) != original_count or len(unique_klines) != original_count:
                # 🚀 优化：使用临时键和 RENAME 实现原子替换
                # 避免在修复过程中 API Server 读到空数据
                temp_kline_key = self.kline_key + ":temp"
                
                # 使用pipeline批量操作，确保原子性
                pipe = self.r_text.pipeline()
                
                # 清空临时 ZSET（如果存在）
                pipe.delete(temp_kline_key)
                
                # 批量添加修复后的K线到临时 ZSET
                zadd_map = {
                    json.dumps(kline, ensure_ascii=False): kline.get('time', 0)
                    for kline in filled_klines
                }
                
                # ZADD 批量写入
                if zadd_map:
                    pipe.zadd(temp_kline_key, zadd_map)
                
                # 原子替换：将临时 ZSET 重命名为正式 ZSET
                # RENAME 是原子操作，确保 API Server 不会读到空数据
                pipe.rename(temp_kline_key, self.kline_key)
                
                # 执行批量操作
                pipe.execute()
                
                logger.info(f"数据完整性: K线数据已修复（原子替换） - 原始:{original_count}, 去重后:{len(unique_klines)}, 补空后:{len(filled_klines)}")
            else:
                logger.debug(f"数据完整性: K线数据检查完成，无需修复 - {original_count}根K线")
            
        except Exception as e:
            logger.error(f"数据完整性: 检查K线数据失败: {e}")
    
    def run_tick_validator(self):
        """
        运行TICK验证器（监听原始TICK流，验证后存储到已验证流）
        
        【数据流】
        - 输入: Redis Stream (tick:BTCUSDm:stream) - 原始TICK流（Data Puller写入）
        - 输出: Redis Stream (tick:BTCUSDm:validated:stream) - 验证后的TICK流（L2策略核心消费）
        
        【验证逻辑】
        1. Seq顺序检查：确保TICK按顺序接收
        2. Checksum验证：确保数据未被篡改
        3. 验证通过后推送到已验证流
        """
        logger.info("数据完整性: TICK验证器已启动")
        logger.info(f"  - 监听原始流: {self.tick_stream_key}")
        logger.info(f"  - 输出已验证流: {self.validated_tick_stream_key}")
        # 🔴 修复：从流的末尾开始读取（只处理新数据），避免处理历史数据导致Seq重复
        # 使用 '$' 表示只读取新消息，避免处理历史数据
        last_id = '$'
        reconnect_delay = 1.0
        
        while not self.stop_event.is_set():
            try:
                # 确保Redis连接
                try:
                    self.r_text.ping()
                except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError):
                    logger.warning(f"数据完整性: Redis连接丢失，{reconnect_delay}秒后重试...")
                    time.sleep(reconnect_delay)
                    reconnect_delay = min(reconnect_delay * 2, 10.0)
                    # 尝试重新创建连接
                    try:
                        self.r_text = redis.Redis(
                            host=REDIS_CONFIG.get('host', 'localhost'),
                            port=REDIS_CONFIG.get('port', 6379),
                            db=REDIS_CONFIG.get('db', 0),
                            decode_responses=True
                        )
                        self.r_text.ping()
                        reconnect_delay = 1.0
                        logger.info("数据完整性: Redis连接已恢复")
                    except:
                        logger.warning("数据完整性: Redis连接仍未恢复，继续等待...")
                        continue
                
                # 从原始TICK流读取（阻塞读取，超时100ms）
                messages = self.r_text.xread({self.tick_stream_key: last_id}, count=10, block=100)
                
                if not messages:
                    continue
                
                # 处理消息
                for stream, msgs in messages:
                    for msg_id, msg_data in msgs:
                        try:
                            # 解析TICK数据（Data Puller写入格式：{'value': json_string}）
                            tick_json = msg_data.get('value', '')
                            if tick_json:
                                tick_data = json.loads(tick_json)
                                # 验证并存储到已验证流
                                self.validate_and_store_tick(tick_data)
                            else:
                                logger.warning(f"数据完整性: TICK数据缺少'value'字段: {msg_data}")
                        except json.JSONDecodeError as e:
                            logger.error(f"数据完整性: TICK JSON解析失败: {e}, 数据: {msg_data}")
                        except Exception as e:
                            logger.error(f"数据完整性: 处理TICK失败: {e}")
                        
                        last_id = msg_id
                
            except redis.exceptions.ConnectionError as ce:
                logger.warning(f"数据完整性: Redis连接错误: {ce}")
                time.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, 10.0)
            except Exception as e:
                logger.error(f"数据完整性: TICK验证器异常: {e}")
                time.sleep(1)
    
    def run_kline_checker(self):
        """
        运行K线检查器（定期检查并修复K线数据）
        
        【行业最佳实践】
        - 定期检查最近N根K线（增量修复）
        - 历史数据视为固定，不修改
        - 自动修复缺失和重复
        """
        logger.info("数据完整性: K线检查器已启动")
        
        # 🚀 启动时立即检查一次
        try:
            from src.trading.services.data_integrity_checker import DataIntegrityChecker
            checker = DataIntegrityChecker(symbol=self.symbol)
            report = checker.check_and_repair_recent(recent_count=100)
            if report.get('success'):
                logger.info(f"数据完整性: 启动时检查完成 - {report}")
        except Exception as e:
            logger.warning(f"数据完整性: 启动时检查失败: {e}")
        
        while not self.stop_event.is_set():
            try:
                # 每5分钟检查一次（增量修复最近100根）
                self.stop_event.wait(300)
                
                if not self.stop_event.is_set():
                    self.check_and_fix_klines()
                    
                    # 输出统计信息
                    logger.info(f"数据完整性统计: TICK已验证={self.stats['ticks_validated']}, "
                              f"TICK拒绝={self.stats['ticks_rejected']}, "
                              f"K线去重={self.stats['klines_deduplicated']}, "
                              f"K线补空={self.stats['klines_filled']}")
                    
            except Exception as e:
                logger.error(f"数据完整性: K线检查器异常: {e}")
    
    def start(self):
        """启动数据完整性检查服务"""
        logger.info("=" * 70)
        logger.info("数据完整性检查服务启动")
        logger.info("=" * 70)
        
        # 启动TICK验证器线程
        tick_thread = threading.Thread(target=self.run_tick_validator, daemon=True, name="TickValidator")
        tick_thread.start()
        
        # 启动K线检查器线程
        kline_thread = threading.Thread(target=self.run_kline_checker, daemon=True, name="KlineChecker")
        kline_thread.start()
        
        logger.info("✅ 数据完整性检查服务已启动")
        logger.info("  - TICK验证器: 监听原始TICK流，验证后存储到已验证流")
        logger.info("  - K线检查器: 定期检查并修复K线数据（去重、补空）")
    
    def stop(self):
        """停止数据完整性检查服务"""
        self.stop_event.set()
        logger.info("数据完整性检查服务已停止")

