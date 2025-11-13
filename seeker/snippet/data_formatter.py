#date: 2025-11-13T17:03:12Z
#url: https://api.github.com/gists/d68fdd03a1f1096971a5f4ed71530eb2
#owner: https://api.github.com/users/wangwei334455

"""
数据格式化器 - 将MT5数据格式化为标准格式

【职责划分】
1. 通用格式化：用于前端展示、图表库（ECharts/lightweight-charts）
2. L2内部格式化：用于HFT策略层，附加序列号和校验和

【设计说明】
- 使用@staticmethod避免实例化，保持简洁性
- JSON序列化使用ensure_ascii=False，支持UTF-8编码
- 浮点数精度：保留6-8位小数，满足HFT精度要求
"""
import logging
import hashlib
import threading
from typing import Dict, List, Optional
import json

logger = logging.getLogger(__name__)


class DataFormatter:
    """
    数据格式化器类
    
    【功能分类】
    1. 通用格式化（前端/图表）：
       - format_kline_standard: 标准K线格式
       - format_tick_standard: 简单TICK格式
       - format_tick_detailed: 详细TICK格式
    
    2. L2策略层格式化（HFT内部）：
       - format_tick_l2_internal: 附加序列号和校验和
    """
    
    # 线程安全的序列号计数器（用于L2内部格式化）
    _sequence_counter = 0
    _sequence_lock = threading.Lock()
    
    @staticmethod
    def format_kline(bar: Dict) -> Dict:
        """
        格式化K线数据（通用格式）
        
        Args:
            bar: MT5 K线数据或字典
            
        Returns:
            格式化后的K线数据
        """
        try:
            volume = bar.get('tick_volume', bar.get('volume', 0))
            real_volume = bar.get('real_volume', 0)
            
            return {
                'time': int(bar['time']),
                'open': float(bar['open']),
                'high': float(bar['high']),
                'low': float(bar['low']),
                'close': float(bar['close']),
                'volume': int(volume),
                'real_volume': int(real_volume),
            }
        except Exception as e:
            logger.error(f"格式化K线数据失败: {str(e)}")
            return None
    
    @staticmethod
    def format_kline_standard(bar: Dict) -> Dict:
        """
        将MT5 K线数据格式化为统一标准格式（MT5KLine）
        
        【统一标准格式】
        {
            "time": 1234567890,  // Unix timestamp (秒, UTC)
            "timezone": "UTC",   // 时区标识
            "open": 1.2345,
            "high": 1.2350,
            "low": 1.2340,
            "close": 1.2348,
            "volume": 12345,     // tick_volume
            "real_volume": 123    // real_volume（实际成交量，可选）
        }
        
        【时区标准化】
        - MT5返回的时间戳是UTC时间
        - 所有时间戳统一存储为UTC Unix时间戳（秒）
        - 添加timezone字段明确标识，避免跨时区或夏令时问题
        
        【数据一致性】
        - 确保前端、后端、中继器、MT5使用相同的数据结构
        - 图表使用时需要转换为图表库格式（通过chartConverter）
        
        Args:
            bar: MT5 K线数据
            
        Returns:
            统一格式的MT5KLine
        """
        try:
            # 导入统一转换函数
            from src.trading.utils.market_data_converter import convert_to_mt5_kline
            return convert_to_mt5_kline(bar)
        except ImportError:
            # 如果导入失败，使用本地实现（向后兼容）
            volume = bar.get('tick_volume', bar.get('volume', 0))
            real_volume = bar.get('real_volume', 0)
            time_utc = int(bar['time'])
            
            return {
                'time': time_utc,
                'timezone': 'UTC',
                'open': float(bar['open']),
                'high': float(bar['high']),
                'low': float(bar['low']),
                'close': float(bar['close']),
                'volume': int(volume),
                'real_volume': int(real_volume),
            }
        except Exception as e:
            logger.error(f"格式化K线数据失败: {str(e)}")
            return None
    
    @staticmethod
    def format_tick_standard(tick) -> Dict:
        """
        将MT5 Tick数据格式化为标准格式
        
        标准TICK格式（适用于ECharts等图表库）:
        {
            "time": 1234567890,  // Unix timestamp (秒)
            "value": 1.2345      // 价格
        }
        
        Args:
            tick: MT5 Tick数据（可能是dict或numpy结构化数组）
            
        Returns:
            格式化后的数据
        """
        try:
            # 处理numpy结构化数组（numpy.void）
            if hasattr(tick, 'dtype') and hasattr(tick, 'time'):
                # 这是numpy结构化数组，直接访问属性
                time_val = int(tick.time)
                last_val = float(getattr(tick, 'last', 0))
                bid_val = float(getattr(tick, 'bid', 0))
                ask_val = float(getattr(tick, 'ask', 0))
            else:
                # 这是普通字典，使用.get()方法
                time_val = int(tick['time'])
                last_val = float(tick.get('last', 0))
                bid_val = float(tick.get('bid', 0))
                ask_val = float(tick.get('ask', 0))
            
            # 使用last价格，如果没有则使用bid和ask的中间价
            price = last_val
            if price == 0:
                price = (bid_val + ask_val) / 2
            
            formatted = {
                'time': time_val,  # Unix时间戳（秒）
                'value': float(price),
            }
            return formatted
        except Exception as e:
            logger.error(f"格式化Tick数据失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    @staticmethod
    def format_tick_detailed(tick) -> Dict:
        """
        格式化详细的Tick数据为统一标准格式（MT5Tick）
        
        【统一标准格式】
        {
            "time": 1234567890,      // Unix timestamp (秒, UTC)
            "time_msc": 1234567890000, // Unix timestamp (毫秒, UTC)
            "timezone": "UTC",        // 时区标识
            "bid": 1.2345,
            "ask": 1.2346,
            "last": 1.23455,
            "volume": 12345
        }
        
        【数据一致性】
        - 确保前端、后端、中继器、MT5使用相同的数据结构
        - 图表使用时需要转换为图表库格式（通过chartConverter）
        
        Args:
            tick: MT5 Tick数据（可能是dict或numpy结构化数组）
            
        Returns:
            统一格式的MT5Tick
        """
        try:
            # 导入统一转换函数
            from src.trading.utils.market_data_converter import convert_to_mt5_tick
            
            # 如果是numpy结构化数组，先转换为字典
            if hasattr(tick, 'dtype') and hasattr(tick, 'time'):
                tick_dict = {
                    'time': int(tick.time),
                    'time_msc': int(getattr(tick, 'time_msc', int(tick.time) * 1000)),
                    'bid': float(getattr(tick, 'bid', 0)),
                    'ask': float(getattr(tick, 'ask', 0)),
                    'last': float(getattr(tick, 'last', 0)),
                    'volume': int(getattr(tick, 'volume', 0)),
                }
                return convert_to_mt5_tick(tick_dict)
            else:
                return convert_to_mt5_tick(tick)
        except ImportError:
            # 如果导入失败，使用本地实现（向后兼容）
            if hasattr(tick, 'dtype') and hasattr(tick, 'time'):
                time_val = int(tick.time)
                time_msc_val = int(getattr(tick, 'time_msc', time_val * 1000))
                bid_val = float(getattr(tick, 'bid', 0))
                ask_val = float(getattr(tick, 'ask', 0))
                last_val = float(getattr(tick, 'last', 0))
                volume_val = int(getattr(tick, 'volume', 0))
            else:
                time_val = int(tick['time'])
                time_msc_val = int(tick.get('time_msc', time_val * 1000))
                bid_val = float(tick.get('bid', 0))
                ask_val = float(tick.get('ask', 0))
                last_val = float(tick.get('last', 0))
                volume_val = int(tick.get('volume', 0))
            
            return {
                'time': time_val,
                'time_msc': time_msc_val,
                'timezone': 'UTC',
                'bid': bid_val,
                'ask': ask_val,
                'last': last_val,
                'volume': volume_val,
            }
        except Exception as e:
            logger.error(f"格式化详细Tick数据失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    @staticmethod
    def _get_next_sequence() -> int:
        """
        获取下一个序列号（线程安全）
        
        Returns:
            序列号
        """
        with DataFormatter._sequence_lock:
            DataFormatter._sequence_counter += 1
            return DataFormatter._sequence_counter
    
    @staticmethod
    def _calculate_checksum(tick_data: Dict) -> str:
        """
        计算TICK数据的校验和
        
        Args:
            tick_data: TICK数据字典
            
        Returns:
            校验和字符串（MD5前8位）
        """
        try:
            # 构建校验字符串（排除seq和checksum字段）
            check_str = f"{tick_data.get('time_msc', 0)}|{tick_data.get('bid', 0)}|{tick_data.get('ask', 0)}|{tick_data.get('last', 0)}|{tick_data.get('volume', 0)}"
            checksum = hashlib.md5(check_str.encode('utf-8')).hexdigest()[:8]
            return checksum
        except Exception as e:
            logger.error(f"计算校验和失败: {str(e)}")
            return "00000000"
    
    @staticmethod
    def format_tick_l2_internal(raw_tick: Dict, sequence: Optional[int] = None, checksum: Optional[str] = None) -> Optional[Dict]:
        """
        L2 HFT策略层专用的详细TICK格式（附加序列号和校验和）
        
        【用途】用于L2策略层内部数据流，确保数据完整性和顺序性
        
        Args:
            raw_tick: MT5原始TICK数据
            sequence: 外部传入的原子序列号（如果为None，则自动生成）
            checksum: 外部计算的校验和（如果为None，则自动计算）
            
        Returns:
            格式化后的TICK数据（包含seq和checksum字段）
        """
        try:
            # 先格式化为详细格式
            formatted = DataFormatter.format_tick_detailed(raw_tick)
            if not formatted:
                return None
            
            # 附加序列号（如果未提供，则自动生成）
            if sequence is None:
                formatted['seq'] = DataFormatter._get_next_sequence()
            else:
                formatted['seq'] = sequence
            
            # 附加校验和（如果未提供，则自动计算）
            if checksum is None:
                formatted['checksum'] = DataFormatter._calculate_checksum(formatted)
            else:
                formatted['checksum'] = checksum
            
            # 标记数据来源
            formatted['source'] = 'MT5'
            
            return formatted
            
        except Exception as e:
            logger.error(f"格式化L2内部TICK数据失败: {str(e)}")
            return None
    
    @staticmethod
    def format_klines_batch(bars: List[Dict]) -> List[Dict]:
        """
        批量格式化K线数据
        
        Args:
            bars: K线数据列表
            
        Returns:
            格式化后的数据列表
        """
        formatted_bars = []
        for bar in bars:
            formatted = DataFormatter.format_kline_standard(bar)
            if formatted:
                formatted_bars.append(formatted)
        
        logger.info(f"批量格式化了 {len(formatted_bars)} 条K线数据")
        return formatted_bars
    
    @staticmethod
    def to_json(data: any) -> str:
        """
        将数据转换为JSON字符串
        
        Args:
            data: 要转换的数据
            
        Returns:
            JSON字符串
        """
        try:
            return json.dumps(data, ensure_ascii=False)
        except Exception as e:
            logger.error(f"转换JSON失败: {str(e)}")
            return None
    
    @staticmethod
    def from_json(json_str: str) -> any:
        """
        从JSON字符串解析数据
        
        Args:
            json_str: JSON字符串
            
        Returns:
            解析后的数据
        """
        try:
            return json.loads(json_str)
        except Exception as e:
            logger.error(f"解析JSON失败: {str(e)}")
            return None
