#date: 2025-11-13T17:02:59Z
#url: https://api.github.com/gists/c1c2c32c02ad6b73f59da7fc63add2f0
#owner: https://api.github.com/users/wangwei334455

"""
数据完整性检查器（行业最佳实践）

【设计原则】
1. 启动时初始化：后端启动时自动从MT5拉取历史数据并补空
2. 定期完整性检查：定期检查K线时间间隔，自动修复
3. 增量修复：只修复最近的数据，历史数据视为固定
4. 数据质量监控：监控缺失率、重复率、时间间隔异常
5. 原子操作：使用Redis事务确保数据一致性

【数据完整性标准】
- 时间间隔：必须严格等于周期（60秒、300秒等）
- 时间连续性：不能有缺失的K线
- 数据唯一性：相同时间戳只能有一条K线
- 数据完整性：OHLCV字段必须完整
"""
import json
import time
import redis
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from loguru import logger

from config.redis_config import REDIS_CONFIG


class DataIntegrityChecker:
    """
    数据完整性检查器（行业最佳实践）
    
    【职责】
    1. 启动时数据初始化：从MT5拉取历史数据并补空
    2. 定期完整性检查：检查K线时间间隔，自动修复
    3. 数据质量监控：统计缺失率、重复率、异常率
    4. 增量修复：只修复最近N根K线，历史数据视为固定
    """
    
    def __init__(self, symbol: str = "BTCUSDm"):
        self.symbol = symbol
        self.redis_client = redis.Redis(
            host=REDIS_CONFIG.get('host', 'localhost'),
            port=REDIS_CONFIG.get('port', 6379),
            db=REDIS_CONFIG.get('db', 0),
            decode_responses=True
        )
        
        self.kline_key = f"kline:{symbol}:1m"
        
        # 数据质量统计
        self.stats = {
            'total_klines': 0,
            'missing_klines': 0,
            'duplicate_klines': 0,
            'invalid_intervals': 0,
            'filled_klines': 0,
        }
        
        logger.info(f"数据完整性检查器初始化: {symbol}")
    
    def check_time_interval(self, klines: List[Dict[str, Any]], timeframe: str = '1m') -> Tuple[List[int], int]:
        """
        检查K线时间间隔（行业最佳实践：严格检查）
        
        Args:
            klines: K线列表（已排序）
            timeframe: 时间周期
            
        Returns:
            (异常位置列表, 缺失K线总数)
        """
        if len(klines) < 2:
            return [], 0
        
        # 时间周期映射（秒）
        timeframe_seconds = {
            '1m': 60, '5m': 300, '15m': 900, '30m': 1800,
            '1h': 3600, '4h': 14400, '1d': 86400
        }
        expected_interval = timeframe_seconds.get(timeframe, 60)
        
        anomalies = []
        total_missing = 0
        
        for i in range(len(klines) - 1):
            prev_time = klines[i].get('time', 0)
            current_time = klines[i + 1].get('time', 0)
            actual_interval = current_time - prev_time
            
            # 严格检查：允许1秒容差（网络延迟、时钟同步）
            if actual_interval != expected_interval:
                if actual_interval > expected_interval + 1:
                    # 缺失K线
                    missing_count = (actual_interval // expected_interval) - 1
                    anomalies.append(i)
                    total_missing += missing_count
                    logger.debug(f"位置{i}: 缺失{missing_count}根K线 (间隔={actual_interval}秒, 期望={expected_interval}秒)")
                elif actual_interval < expected_interval - 1:
                    # 异常间隔（可能是重复或错误数据）
                    anomalies.append(i)
                    logger.warning(f"位置{i}: 异常间隔={actual_interval}秒 (期望={expected_interval}秒)")
        
        return anomalies, total_missing
    
    def fill_missing_klines(self, klines: List[Dict[str, Any]], timeframe: str = '1m') -> List[Dict[str, Any]]:
        """
        填充缺失的K线（行业最佳实践：精确计算）
        
        Args:
            klines: K线列表（已排序）
            timeframe: 时间周期
            
        Returns:
            填充后的K线列表
        """
        if len(klines) < 2:
            return klines
        
        timeframe_seconds = {
            '1m': 60, '5m': 300, '15m': 900, '30m': 1800,
            '1h': 3600, '4h': 14400, '1d': 86400
        }
        expected_interval = timeframe_seconds.get(timeframe, 60)
        
        filled_klines = [klines[0]]
        
        for i in range(1, len(klines)):
            prev_kline = klines[i - 1]
            current_kline = klines[i]
            
            prev_time = prev_kline.get('time', 0)
            current_time = current_kline.get('time', 0)
            actual_interval = current_time - prev_time
            
            # 如果时间间隔大于期望间隔，填充缺失的K线
            if actual_interval > expected_interval + 1:
                kline_count_in_gap = actual_interval // expected_interval
                missing_count = kline_count_in_gap - 1
                
                if missing_count >= 1:
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
                            self.stats['filled_klines'] += 1
            
            filled_klines.append(current_kline)
        
        return filled_klines
    
    def deduplicate_klines(self, klines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        去重K线数据（相同时间戳只保留最新的）
        
        Args:
            klines: K线列表
            
        Returns:
            去重后的K线列表
        """
        seen_times = {}
        duplicates = 0
        
        for kline in klines:
            kline_time = kline.get('time', 0)
            if kline_time > 0:
                if kline_time in seen_times:
                    # 保留最新的数据（volume更大的）
                    if kline.get('volume', 0) > seen_times[kline_time].get('volume', 0):
                        seen_times[kline_time] = kline
                        duplicates += 1
                else:
                    seen_times[kline_time] = kline
        
        self.stats['duplicate_klines'] = duplicates
        
        unique_klines = list(seen_times.values())
        unique_klines.sort(key=lambda x: x.get('time', 0))
        
        return unique_klines
    
    def initialize_from_mt5(self, count: int = 2880, timeframe: str = '1m') -> bool:
        """
        启动时从MT5初始化历史数据（行业最佳实践：启动时自动拉取）
        
        Args:
            count: 拉取数量（默认2880根，即2天M1数据）
            timeframe: 时间周期
            
        Returns:
            是否成功
        """
        try:
            from src.trading.services.grpc_trade_client import get_grpc_client, is_grpc_available
            
            if not is_grpc_available():
                logger.warning("gRPC不可用，跳过MT5历史数据初始化")
                return False
            
            logger.info(f"🚀 开始从MT5拉取历史K线数据: {self.symbol} {timeframe} x {count}")
            
            client = get_grpc_client()
            client.timeout = 10  # 10秒超时
            
            # 计算时间范围
            try:
                import pytz
                timezone = pytz.timezone("Etc/UTC")
                to_dt = datetime.now(timezone)
            except ImportError:
                # 如果没有pytz，使用UTC时间
                to_dt = datetime.utcnow()
            
            timeframe_minutes = {
                '1m': 1, '5m': 5, '15m': 15, '30m': 30,
                '1h': 60, '4h': 240, '1d': 1440
            }
            minutes_per_kline = timeframe_minutes.get(timeframe, 1)
            from_dt = to_dt - timedelta(minutes=count * minutes_per_kline)
            
            to_time = int(to_dt.timestamp())
            from_time = int(from_dt.timestamp())
            
            result = client.get_klines(
                symbol=self.symbol,
                timeframe=timeframe,
                from_time=from_time,
                to_time=to_time,
                count=count
            )
            
            if result.get('success') and result.get('klines'):
                mt5_klines = result['klines']
                logger.info(f"✓ 从MT5获取到 {len(mt5_klines)} 根历史K线")
                
                # 转换为标准格式
                klines = []
                for k in mt5_klines:
                    klines.append({
                        'time': int(k['time']),
                        'open': float(k['open']),
                        'high': float(k['high']),
                        'low': float(k['low']),
                        'close': float(k['close']),
                        'volume': int(k.get('volume', k.get('tick_volume', 0))),
                        'real_volume': int(k.get('real_volume', 0))
                    })
                
                # 1. 去重
                unique_klines = self.deduplicate_klines(klines)
                
                # 2. 按时间排序
                unique_klines.sort(key=lambda x: x.get('time', 0))
                
                # 3. 补空
                filled_klines = self.fill_missing_klines(unique_klines, timeframe)
                
                # 4. 存储到Redis（原子操作）
                self._store_klines_atomic(filled_klines)
                
                logger.info(f"✅ 历史数据初始化完成: 原始={len(klines)}, 去重后={len(unique_klines)}, 补空后={len(filled_klines)}")
                return True
            else:
                logger.warning(f"从MT5获取历史数据失败: {result.get('message', '未知错误')}")
                return False
                
        except Exception as e:
            logger.error(f"初始化历史数据失败: {e}", exc_info=True)
            return False
    
    def _store_klines_atomic(self, klines: List[Dict[str, Any]]):
        """
        原子存储K线数据（使用临时键+RENAME）
        
        Args:
            klines: K线列表
        """
        temp_key = self.kline_key + ":temp"
        
        pipe = self.redis_client.pipeline()
        pipe.delete(temp_key)
        
        # 批量添加
        zadd_map = {
            json.dumps(kline, ensure_ascii=False): kline.get('time', 0)
            for kline in klines
        }
        
        if zadd_map:
            pipe.zadd(temp_key, zadd_map)
        
        # 原子替换
        pipe.rename(temp_key, self.kline_key)
        pipe.execute()
    
    def check_and_repair_recent(self, recent_count: int = 100, timeframe: str = '1m') -> Dict[str, Any]:
        """
        检查并修复最近的K线数据（增量修复，行业最佳实践）
        
        【策略】
        - 只检查最近N根K线（默认100根）
        - 历史数据视为固定，不修改
        - 自动修复缺失和重复
        
        Args:
            recent_count: 检查最近N根K线
            timeframe: 时间周期
            
        Returns:
            修复统计信息
        """
        try:
            # 从Redis读取最近N根K线
            klines_json = self.redis_client.zrange(
                self.kline_key, -recent_count, -1, withscores=False
            )
            
            if not klines_json:
                return {'success': False, 'message': '没有K线数据'}
            
            # 解析K线数据
            klines = [json.loads(k) for k in klines_json]
            original_count = len(klines)
            
            # 1. 去重
            unique_klines = self.deduplicate_klines(klines)
            
            # 2. 按时间排序
            unique_klines.sort(key=lambda x: x.get('time', 0))
            
            # 3. 检查时间间隔
            anomalies, missing_count = self.check_time_interval(unique_klines, timeframe)
            self.stats['invalid_intervals'] = len(anomalies)
            self.stats['missing_klines'] = missing_count
            
            # 4. 补空
            filled_klines = self.fill_missing_klines(unique_klines, timeframe)
            
            # 5. 如果有变化，更新Redis（只更新最近的数据）
            if len(filled_klines) != original_count or len(unique_klines) != original_count:
                # 删除最近的数据
                if klines:
                    first_time = klines[0].get('time', 0)
                    last_time = klines[-1].get('time', 0)
                    if first_time > 0 and last_time > 0:
                        self.redis_client.zremrangebyscore(
                            self.kline_key, first_time, last_time
                        )
                
                # 添加修复后的数据
                for kline in filled_klines:
                    kline_json = json.dumps(kline, ensure_ascii=False)
                    kline_time = kline.get('time', 0)
                    self.redis_client.zadd(self.kline_key, {kline_json: kline_time})
                
                logger.info(f"✅ 最近数据修复完成: 原始={original_count}, 去重后={len(unique_klines)}, 补空后={len(filled_klines)}")
            
            self.stats['total_klines'] = len(filled_klines)
            
            return {
                'success': True,
                'original_count': original_count,
                'unique_count': len(unique_klines),
                'filled_count': len(filled_klines),
                'missing_count': missing_count,
                'duplicate_count': self.stats['duplicate_klines'],
                'anomalies': len(anomalies)
            }
            
        except Exception as e:
            logger.error(f"检查并修复数据失败: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}
    
    def get_quality_report(self) -> Dict[str, Any]:
        """
        获取数据质量报告
        
        Returns:
            数据质量统计信息
        """
        try:
            total_count = self.redis_client.zcard(self.kline_key)
            
            if total_count == 0:
                return {
                    'total_count': 0,
                    'status': 'empty',
                    'message': '没有K线数据'
                }
            
            # 检查最近100根K线
            recent_report = self.check_and_repair_recent(recent_count=100)
            
            # 计算质量指标
            missing_rate = (self.stats['missing_klines'] / max(total_count, 1)) * 100
            duplicate_rate = (self.stats['duplicate_klines'] / max(total_count, 1)) * 100
            
            quality_score = 100 - missing_rate - duplicate_rate - (len(recent_report.get('anomalies', [])) * 0.1)
            quality_score = max(0, min(100, quality_score))
            
            return {
                'total_count': total_count,
                'missing_count': self.stats['missing_klines'],
                'duplicate_count': self.stats['duplicate_klines'],
                'filled_count': self.stats['filled_klines'],
                'invalid_intervals': self.stats['invalid_intervals'],
                'missing_rate': round(missing_rate, 2),
                'duplicate_rate': round(duplicate_rate, 2),
                'quality_score': round(quality_score, 2),
                'status': 'good' if quality_score >= 95 else 'warning' if quality_score >= 80 else 'poor',
                'recent_report': recent_report
            }
            
        except Exception as e:
            logger.error(f"获取数据质量报告失败: {e}")
            return {'success': False, 'error': str(e)}


def initialize_data_on_startup(symbol: str = "BTCUSDm", count: int = 2880) -> bool:
    """
    启动时数据初始化（行业最佳实践：后端启动时自动调用）
    
    Args:
        symbol: 交易品种
        count: 拉取数量
        
    Returns:
        bool: 是否成功
    """
    checker = DataIntegrityChecker(symbol)
    
    # 检查Redis中是否已有数据
    existing_count = checker.redis_client.zcard(checker.kline_key)
    
    if existing_count == 0:
        logger.info("Redis中没有K线数据，开始从MT5初始化...")
        success = checker.initialize_from_mt5(count=count)
        if success:
            logger.info("✅ 启动时数据初始化成功")
            return True
        else:
            logger.warning("⚠️ 启动时数据初始化失败，将使用实时数据构建")
            return False
    else:
        logger.info(f"Redis中已有{existing_count}根K线，跳过初始化")
        
        # 检查并修复最近的数据
        report = checker.check_and_repair_recent(recent_count=100)
        if report.get('success'):
            logger.info(f"✅ 最近数据检查完成: {report}")
        
        return True  # 已有数据，视为成功

