#date: 2025-11-13T17:03:12Z
#url: https://api.github.com/gists/d68fdd03a1f1096971a5f4ed71530eb2
#owner: https://api.github.com/users/wangwei334455

"""
市场日历和交易时段管理
用于区分正常休市和数据采集异常
"""
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import MetaTrader5 as mt5


class MarketCalendar:
    """
    市场日历管理器
    
    【官方最佳实践】：
    1. 维护交易所日历
    2. 区分休市和采集错误
    3. 记录特殊事件（节假日、服务器维护）
    """
    
    def __init__(self, symbol: str = 'BTCUSDm'):
        self.symbol = symbol
        self.special_events = {}  # 特殊事件记录（节假日、维护等）
    
    def is_market_open(self, check_time: Optional[datetime] = None) -> Dict[str, Any]:
        """
        检查市场是否开市
        
        【核心逻辑】：
        1. BTC现货：24/7交易
        2. BTC CFD：可能有周末维护窗口
        3. 通过MT5 API实时查询市场状态
        
        Returns:
            dict: {
                'is_open': bool,           # 是否开市
                'reason': str,             # 如果闭市，原因
                'next_open': datetime,     # 下次开市时间
                'session_info': dict       # 交易时段信息
            }
        """
        if check_time is None:
            check_time = datetime.now()
        
        result = {
            'is_open': True,
            'reason': '',
            'next_open': None,
            'session_info': {},
            'check_time': check_time
        }
        
        try:
            # 【方法1】通过MT5查询交易品种状态（最准确）
            if mt5.symbol_info(self.symbol) is None:
                # 连接MT5
                if not mt5.initialize():
                    result['is_open'] = False
                    result['reason'] = 'MT5未连接'
                    return result
            
            symbol_info = mt5.symbol_info(self.symbol)
            
            if symbol_info is None:
                result['is_open'] = False
                result['reason'] = f'交易品种{self.symbol}不可用'
                return result
            
            # 检查交易品种是否可见和可交易
            if not symbol_info.visible:
                result['is_open'] = False
                result['reason'] = '交易品种不可见'
                return result
            
            # 获取交易时段信息
            result['session_info'] = {
                'trade_mode': symbol_info.trade_mode,  # 交易模式
                'trade_stops_level': symbol_info.trade_stops_level,
                'volume_min': symbol_info.volume_min,
                'volume_max': symbol_info.volume_max,
            }
            
            # 【方法2】检查是否在特殊事件期间
            if self._is_special_event(check_time):
                result['is_open'] = False
                result['reason'] = '特殊事件期间（节假日/维护）'
                result['next_open'] = self._get_next_open_time(check_time)
                return result
            
            # 【方法3】尝试获取最新报价（最可靠的方法）
            tick = mt5.symbol_info_tick(self.symbol)
            if tick is None:
                result['is_open'] = False
                result['reason'] = '无法获取报价（可能休市或服务器维护）'
                return result
            
            # 检查报价时间（如果报价时间超过5分钟，认为市场可能闭市）
            tick_time = datetime.fromtimestamp(tick.time)
            time_diff = (datetime.now() - tick_time).total_seconds()
            
            if time_diff > 300:  # 超过5分钟
                result['is_open'] = False
                result['reason'] = f'报价过期（{int(time_diff/60)}分钟前）'
                return result
            
            # 市场开市
            result['is_open'] = True
            result['reason'] = '市场正常交易'
            result['session_info']['last_tick_time'] = tick_time
            result['session_info']['bid'] = tick.bid
            result['session_info']['ask'] = tick.ask
            
        except Exception as e:
            result['is_open'] = False
            result['reason'] = f'检查异常: {str(e)}'
        
        return result
    
    def _is_special_event(self, check_time: datetime) -> bool:
        """检查是否在特殊事件期间"""
        date_key = check_time.strftime('%Y-%m-%d')
        return date_key in self.special_events
    
    def _get_next_open_time(self, current_time: datetime) -> Optional[datetime]:
        """获取下次开市时间"""
        # 简化实现：假设下个工作日开市
        next_day = current_time + timedelta(days=1)
        next_day = next_day.replace(hour=0, minute=0, second=0, microsecond=0)
        return next_day
    
    def add_special_event(self, date: str, event_type: str, description: str):
        """
        添加特殊事件
        
        Args:
            date: 日期 (YYYY-MM-DD)
            event_type: 事件类型 (holiday, maintenance, etc.)
            description: 描述
        """
        self.special_events[date] = {
            'type': event_type,
            'description': description
        }
    
    def should_have_data(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """
        【核心方法】判断指定时间段是否应该有数据
        
        用于区分：
        1. 正常休市 → 不应该有数据 → 空位正常
        2. 市场开市 → 应该有数据 → 空位异常（采集错误）
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
        
        Returns:
            dict: {
                'should_have_data': bool,      # 是否应该有数据
                'reason': str,                  # 原因
                'open_periods': list,           # 开市时段列表
                'closed_periods': list,         # 闭市时段列表
                'coverage_ratio': float,        # 开市覆盖率 (0-1)
            }
        """
        result = {
            'should_have_data': True,
            'reason': '',
            'open_periods': [],
            'closed_periods': [],
            'coverage_ratio': 1.0
        }
        
        # 对于BTC CFD，假设24/7交易，除非在特殊事件期间
        total_seconds = (end_time - start_time).total_seconds()
        open_seconds = 0
        
        # 按小时检查
        current = start_time
        while current < end_time:
            market_status = self.is_market_open(current)
            
            next_hour = min(current + timedelta(hours=1), end_time)
            period_seconds = (next_hour - current).total_seconds()
            
            if market_status['is_open']:
                result['open_periods'].append({
                    'start': current,
                    'end': next_hour
                })
                open_seconds += period_seconds
            else:
                result['closed_periods'].append({
                    'start': current,
                    'end': next_hour,
                    'reason': market_status['reason']
                })
            
            current = next_hour
        
        # 计算覆盖率
        if total_seconds > 0:
            result['coverage_ratio'] = open_seconds / total_seconds
        
        # 如果开市时段超过50%，认为应该有数据
        if result['coverage_ratio'] > 0.5:
            result['should_have_data'] = True
            result['reason'] = f'大部分时间开市（{result["coverage_ratio"]*100:.1f}%）'
        else:
            result['should_have_data'] = False
            result['reason'] = f'大部分时间闭市（开市仅{result["coverage_ratio"]*100:.1f}%）'
        
        return result
    
    def validate_data_gap(self, gap_start: datetime, gap_end: datetime, 
                         collector_running: bool = True) -> Dict[str, Any]:
        """
        【核心方法】验证数据缺口的合理性
        
        Args:
            gap_start: 缺口开始时间
            gap_end: 缺口结束时间
            collector_running: 采集器是否在运行
        
        Returns:
            dict: {
                'is_valid': bool,          # 缺口是否合理
                'gap_type': str,           # 缺口类型: normal_closed, collector_offline, data_error
                'should_fill': bool,       # 是否应该填补
                'reason': str,             # 原因说明
                'severity': str,           # 严重程度: low, medium, high, critical
            }
        """
        result = {
            'is_valid': True,
            'gap_type': 'unknown',
            'should_fill': False,
            'reason': '',
            'severity': 'low'
        }
        
        # 检查该时段是否应该有数据
        market_status = self.should_have_data(gap_start, gap_end)
        
        gap_minutes = (gap_end - gap_start).total_seconds() / 60
        
        # 【情况1】市场闭市，缺口正常
        if not market_status['should_have_data']:
            result['is_valid'] = True
            result['gap_type'] = 'normal_closed'
            result['should_fill'] = False
            result['reason'] = f'正常休市: {market_status["reason"]}'
            result['severity'] = 'low'
            return result
        
        # 【情况2】市场开市，但采集器离线
        if not collector_running:
            result['is_valid'] = True  # 缺口合理（已知原因）
            result['gap_type'] = 'collector_offline'
            result['should_fill'] = True  # 应该尝试填补
            result['reason'] = f'采集器离线期间（{gap_minutes:.1f}分钟）'
            result['severity'] = 'high' if gap_minutes > 60 else 'medium'
            return result
        
        # 【情况3】市场开市且采集器运行，但有缺口 → 数据异常
        result['is_valid'] = False
        result['gap_type'] = 'data_error'
        result['should_fill'] = True
        result['reason'] = f'数据采集异常（市场开市但无数据，{gap_minutes:.1f}分钟）'
        
        # 根据缺口大小判断严重程度
        if gap_minutes > 60:
            result['severity'] = 'critical'
        elif gap_minutes > 10:
            result['severity'] = 'high'
        elif gap_minutes > 3:
            result['severity'] = 'medium'
        else:
            result['severity'] = 'low'
        
        return result


def get_market_calendar(symbol: str = 'BTCUSDm') -> MarketCalendar:
    """获取市场日历单例"""
    if not hasattr(get_market_calendar, '_instance'):
        get_market_calendar._instance = MarketCalendar(symbol)
    return get_market_calendar._instance


