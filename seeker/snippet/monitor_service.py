#date: 2025-11-13T17:02:59Z
#url: https://api.github.com/gists/c1c2c32c02ad6b73f59da7fc63add2f0
#owner: https://api.github.com/users/wangwei334455

"""
HFT系统监控与告警服务

【职责】
1. 数据健康监控（TICK延迟、Seq跳跃、数据流中断）
2. 系统健康监控（Redis健康、FSM循环时间）
3. 策略表现监控（订单成交率、策略最大回撤）
4. 实时告警（日志、外部通知）
"""
import time
import threading
import logging
from typing import Dict, Any, List, Optional
from collections import deque
from enum import Enum

logger = logging.getLogger(__name__)


# 告警级别
class AlertLevel(Enum):
    """告警级别定义"""
    INFO = 'INFO'
    WARNING = 'WARNING'
    ERROR = 'ERROR'
    CRITICAL = 'CRITICAL'
    FATAL = 'FATAL'


# 告警阈值配置
class MonitorThresholds:
    """监控阈值配置"""
    # 数据健康
    TICK_LATENCY_MS_MAX = 50.0  # TICK采集到FSM决策的最大允许延迟（毫秒）
    TICK_STREAM_TIMEOUT_SEC = 10.0  # TICK流中断超时（秒）
    SEQ_JUMP_THRESHOLD = 10  # Seq跳跃阈值（超过此值视为异常）
    
    # 系统健康
    FSM_LOOP_TIME_MS_MAX = 1.0  # FSM处理单个TICK的最大时间（毫秒）
    REDIS_PING_MS_MAX = 5.0  # Redis最大ping延迟（毫秒）
    
    # 策略表现
    MAX_DAILY_DRAWDOWN_USD = 500.0  # 最大日内亏损（USD）
    MIN_FILL_RATE_PCT = 90.0  # 最小订单成交率（百分比）
    MAX_ORDER_REJECT_RATE_PCT = 10.0  # 最大订单拒绝率（百分比）


class MonitorService:
    """
    HFT系统的实时监控与告警服务
    
    【监控维度】
    1. 数据健康：TICK延迟、数据流完整性
    2. 系统健康：组件性能、连接状态
    3. 策略表现：交易效率、风险指标
    """
    
    def __init__(self, symbol: str, config: Optional[Dict[str, Any]] = None):
        """
        初始化监控服务
        
        Args:
            symbol: 交易品种
            config: 监控配置字典
        """
        self.symbol = symbol
        self.config = config or {}
        
        # 实时状态存储（用于计算波动指标）
        self.tick_latencies = deque(maxlen=1000)  # 存储最近1000个延迟
        self.fsm_loop_times = deque(maxlen=1000)  # 存储最近1000个FSM循环时间
        self.last_tick_time = time.time()  # 记录上次TICK时间
        self.last_tick_seq = 0  # 记录上次TICK的Seq
        
        # 订单统计
        self.order_stats = {
            'total_orders': 0,
            'filled_orders': 0,
            'rejected_orders': 0,
            'canceled_orders': 0,
            'error_orders': 0
        }
        self.order_stats_lock = threading.Lock()
        
        # 告警历史（防止重复告警）
        self.alert_history = deque(maxlen=100)
        self.alert_lock = threading.Lock()
        
        # 外部通知器（可选）
        self.notifier = None
        # try:
        #     from utils.notifier import Notifier
        #     self.notifier = Notifier()
        # except ImportError:
        #     logger.warning("Notifier未找到，告警将仅记录到日志")
        
        logger.info(f"监控服务已初始化: {symbol}")
    
    # ----------------------------------------
    # I. 数据健康监控
    # ----------------------------------------
    
    def report_tick_health(self, tick: Dict[str, Any], current_fsm_time: float):
        """
        由StrategyFSM调用，报告TICK的延迟和顺序
        
        【关键功能】
        1. 计算端到端延迟（采集时间 -> FSM接收时间）
        2. 心跳检测（数据流中断检测）
        3. Seq跳跃检测（数据完整性）
        
        Args:
            tick: 原始TICK数据（包含time_msc - 采集时间戳）
            current_fsm_time: FSM接收并准备处理TICK的时间戳（秒级）
        """
        try:
            tick_collect_time_ms = tick.get('time_msc', 0)  # 采集时间（毫秒级Unix时间戳）
            tick_seq = tick.get('seq', 0)
            
            # 1. 计算总延迟（采集 -> FSM接收）
            # 注意：time_msc是毫秒级，current_fsm_time是秒级
            latency_ms = current_fsm_time * 1000.0 - tick_collect_time_ms
            
            with self.alert_lock:
                # 记录延迟
                self.tick_latencies.append(latency_ms)
                
                # A. 延迟阈值告警
                if latency_ms > MonitorThresholds.TICK_LATENCY_MS_MAX:
                    self._trigger_alert(
                        level=AlertLevel.CRITICAL,
                        area='DATA_LATENCY',
                        message=f"TICK延迟超限：{latency_ms:.2f} ms > {MonitorThresholds.TICK_LATENCY_MS_MAX} ms。数据流可能阻塞！"
                    )
                
                # B. 心跳检测（检查数据流是否中断）
                time_since_last_tick = current_fsm_time - self.last_tick_time
                if time_since_last_tick > MonitorThresholds.TICK_STREAM_TIMEOUT_SEC:
                    self._trigger_alert(
                        level=AlertLevel.CRITICAL,
                        area='DATA_FLOW_STALL',
                        message=f"数据流中断：上次TICK已间隔 {time_since_last_tick:.1f} 秒，超过阈值 {MonitorThresholds.TICK_STREAM_TIMEOUT_SEC} 秒。"
                    )
                
                self.last_tick_time = current_fsm_time
                
                # C. 检查序列号跳跃
                if self.last_tick_seq > 0:
                    seq_jump = tick_seq - self.last_tick_seq
                    if seq_jump > MonitorThresholds.SEQ_JUMP_THRESHOLD:
                        self._trigger_alert(
                            level=AlertLevel.ERROR,
                            area='DATA_INTEGRITY',
                            message=f"Seq跳跃异常：从 {self.last_tick_seq} 跳到 {tick_seq}，跳跃 {seq_jump}。可能丢失数据！"
                        )
                
                self.last_tick_seq = tick_seq
            
        except Exception as e:
            logger.error(f"报告TICK健康状态失败: {e}")
    
    # ----------------------------------------
    # II. 系统健康监控
    # ----------------------------------------
    
    def report_fsm_performance(self, loop_duration_ms: float):
        """
        由StrategyFSM调用，报告其主循环的处理耗时
        
        【关键功能】
        1. 记录FSM处理单个TICK的耗时
        2. 检测性能瓶颈（可能导致TICK堆积）
        
        Args:
            loop_duration_ms: FSM处理单个TICK的耗时（毫秒）
        """
        try:
            with self.alert_lock:
                # 记录循环时间
                self.fsm_loop_times.append(loop_duration_ms)
                
                # 检查性能阈值
                if loop_duration_ms > MonitorThresholds.FSM_LOOP_TIME_MS_MAX:
                    self._trigger_alert(
                        level=AlertLevel.WARNING,
                        area='FSM_PERFORMANCE',
                        message=f"FSM处理耗时过长：{loop_duration_ms:.3f} ms，超过阈值 {MonitorThresholds.FSM_LOOP_TIME_MS_MAX} ms。可能导致TICK堆积。"
                    )
                    
        except Exception as e:
            logger.error(f"报告FSM性能失败: {e}")
    
    def check_redis_health(self, redis_client) -> bool:
        """
        检查Redis的连接状态和延迟
        
        Args:
            redis_client: Redis客户端实例
            
        Returns:
            bool: Redis是否健康
        """
        try:
            start = time.time()
            redis_client.ping()
            ping_latency_ms = (time.time() - start) * 1000.0
            
            if ping_latency_ms > MonitorThresholds.REDIS_PING_MS_MAX:
                self._trigger_alert(
                    level=AlertLevel.ERROR,
                    area='REDIS_HEALTH',
                    message=f"Redis Ping延迟过高：{ping_latency_ms:.2f} ms，超过阈值 {MonitorThresholds.REDIS_PING_MS_MAX} ms。数据存储/分发可能受影响。"
                )
                return False
            
            return True
            
        except Exception as e:
            self._trigger_alert(
                level=AlertLevel.CRITICAL,
                area='REDIS_HEALTH',
                message=f"Redis连接失败：{e}。系统功能已停止。"
            )
            return False
    
    # ----------------------------------------
    # III. 策略表现与风险监控
    # ----------------------------------------
    
    def report_order_result(self, order_status: str):
        """
        报告订单结果（用于统计成交率）
        
        Args:
            order_status: 订单状态（FILLED, REJECTED, CANCELED, ERROR）
        """
        try:
            with self.order_stats_lock:
                self.order_stats['total_orders'] += 1
                
                if order_status == 'FILLED':
                    self.order_stats['filled_orders'] += 1
                elif order_status == 'REJECTED':
                    self.order_stats['rejected_orders'] += 1
                elif order_status == 'CANCELED':
                    self.order_stats['canceled_orders'] += 1
                elif order_status == 'ERROR':
                    self.order_stats['error_orders'] += 1
                
                # 检查成交率
                if self.order_stats['total_orders'] >= 10:  # 至少10笔订单才开始检查
                    fill_rate = (self.order_stats['filled_orders'] / self.order_stats['total_orders']) * 100
                    if fill_rate < MonitorThresholds.MIN_FILL_RATE_PCT:
                        self._trigger_alert(
                            level=AlertLevel.WARNING,
                            area='TRADE_EFFICIENCY',
                            message=f"订单成交率过低：{fill_rate:.2f}%，低于阈值 {MonitorThresholds.MIN_FILL_RATE_PCT}%。交易所连接或风控设置可能有问题。"
                        )
                    
                    # 检查拒绝率
                    reject_rate = (self.order_stats['rejected_orders'] / self.order_stats['total_orders']) * 100
                    if reject_rate > MonitorThresholds.MAX_ORDER_REJECT_RATE_PCT:
                        self._trigger_alert(
                            level=AlertLevel.ERROR,
                            area='TRADE_EFFICIENCY',
                            message=f"订单拒绝率过高：{reject_rate:.2f}%，超过阈值 {MonitorThresholds.MAX_ORDER_REJECT_RATE_PCT}%。风控或交易所可能有问题。"
                        )
                        
        except Exception as e:
            logger.error(f"报告订单结果失败: {e}")
    
    def report_risk_breach(self, breach_type: str, value: float, limit: float):
        """
        接收来自RiskController的风险突破告警
        
        【关键功能】
        1. 最大回撤突破告警
        2. 最大仓位超限告警
        
        Args:
            breach_type: 突破类型 ('MAX_DD' 最大回撤, 'MAX_POS' 最大仓位)
            value: 当前值
            limit: 限制值
        """
        try:
            if breach_type == 'MAX_DD':
                self._trigger_alert(
                    level=AlertLevel.FATAL,
                    area='RISK_BREACH',
                    message=f"⚠️ 最大日内回撤触发！当前亏损 {value:.2f} USD，已达硬性限制 {limit:.2f} USD。系统应发送紧急平仓指令！"
                )
            elif breach_type == 'MAX_POS':
                self._trigger_alert(
                    level=AlertLevel.CRITICAL,
                    area='RISK_BREACH',
                    message=f"最大仓位超限：当前仓位 {value:.4f} > 限制 {limit:.4f}。"
                )
            elif breach_type == 'FREQUENCY':
                self._trigger_alert(
                    level=AlertLevel.WARNING,
                    area='RISK_BREACH',
                    message=f"交易频率超限：过去1分钟内已有 {int(value)} 笔交易，超过限制 {int(limit)} 笔。"
                )
                
        except Exception as e:
            logger.error(f"报告风险突破失败: {e}")
    
    def report_daily_pnl(self, daily_pnl: float):
        """
        报告日内盈亏（用于风险监控，兼容旧接口）
        
        Args:
            daily_pnl: 日内盈亏（USD）
        """
        # 转换为新的接口格式
        if daily_pnl <= -MonitorThresholds.MAX_DAILY_DRAWDOWN_USD:
            self.report_risk_breach('MAX_DD', daily_pnl, MonitorThresholds.MAX_DAILY_DRAWDOWN_USD)
        elif daily_pnl <= -MonitorThresholds.MAX_DAILY_DRAWDOWN_USD * 0.8:
            self._trigger_alert(
                level=AlertLevel.CRITICAL,
                area='RISK_BREACH',
                message=f"日内亏损接近上限：当前亏损 {daily_pnl:.2f} USD，已达到阈值的80%。"
            )
    
    # ----------------------------------------
    # IV. 告警发送核心
    # ----------------------------------------
    
    def _trigger_alert(self, level: AlertLevel, area: str, message: str):
        """
        发送告警到日志和外部通知渠道
        
        Args:
            level: 告警级别
            area: 告警区域
            message: 告警消息
        """
        try:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            alert_key = f"{area}:{message[:50]}"  # 用于去重
            
            # 检查是否已发送过相同告警（防止重复告警）
            with self.alert_lock:
                if alert_key in self.alert_history:
                    return  # 已发送过，跳过
                self.alert_history.append(alert_key)
            
            alert_message = f"[{level.value} | {timestamp} | {self.symbol} | {area}] {message}"
            
            # 记录到本地关键日志
            if level in [AlertLevel.CRITICAL, AlertLevel.FATAL]:
                logger.error(alert_message)
            elif level == AlertLevel.ERROR:
                logger.error(alert_message)
            elif level == AlertLevel.WARNING:
                logger.warning(alert_message)
            else:
                logger.info(alert_message)
            
            # 发送到外部告警系统（Notifier）
            if self.notifier:
                try:
                    self.notifier.send_notification(alert_message, level.value)
                except Exception as e:
                    logger.error(f"发送外部告警失败: {e}")
                    
        except Exception as e:
            logger.error(f"触发告警失败: {e}")
    
    # ----------------------------------------
    # V. 监控状态查询
    # ----------------------------------------
    
    def get_monitor_status(self) -> Dict[str, Any]:
        """
        获取当前监控状态
        
        Returns:
            监控状态字典
        """
        try:
            # 计算平均延迟
            avg_latency = 0.0
            if self.tick_latencies:
                avg_latency = sum(self.tick_latencies) / len(self.tick_latencies)
            
            # 计算平均FSM循环时间
            avg_loop_time = 0.0
            if self.fsm_loop_times:
                avg_loop_time = sum(self.fsm_loop_times) / len(self.fsm_loop_times)
            
            # 计算成交率
            fill_rate = 0.0
            with self.order_stats_lock:
                if self.order_stats['total_orders'] > 0:
                    fill_rate = (self.order_stats['filled_orders'] / self.order_stats['total_orders']) * 100
            
            return {
                'symbol': self.symbol,
                'tick_health': {
                    'avg_latency_ms': avg_latency,
                    'last_tick_time': self.last_tick_time,
                    'last_tick_seq': self.last_tick_seq
                },
                'system_health': {
                    'avg_fsm_loop_time_ms': avg_loop_time,
                    'fsm_loop_count': len(self.fsm_loop_times)
                },
                'trade_stats': {
                    'total_orders': self.order_stats['total_orders'],
                    'filled_orders': self.order_stats['filled_orders'],
                    'rejected_orders': self.order_stats['rejected_orders'],
                    'fill_rate_pct': fill_rate
                }
            }
            
        except Exception as e:
            logger.error(f"获取监控状态失败: {e}")
            return {}


# --- 系统集成示例 ---
if __name__ == "__main__":
    # 模拟监控服务
    monitor = MonitorService(symbol="BTCUSDm")
    
    # 模拟数据流中断
    current_time = time.time()
    monitor.report_tick_health(
        tick={'time': current_time - 15, 'time_msc': (current_time - 15) * 1000, 'seq': 100},
        current_fsm_time=current_time
    )
    
    # 下次调用会触发中断告警
    time.sleep(1)
    monitor.report_tick_health(
        tick={'time': current_time - 0.06, 'time_msc': (current_time - 0.06) * 1000, 'seq': 101},
        current_fsm_time=time.time()
    )
    
    # 模拟TICK延迟过高
    monitor.report_tick_health(
        tick={'time': current_time - 0.1, 'time_msc': (current_time - 0.1) * 1000, 'seq': 102},
        current_fsm_time=time.time()  # 延迟100ms
    )
    
    # 模拟FSM性能下降
    monitor.report_fsm_performance(loop_duration_ms=5.5)  # 超过1.0ms阈值
    
    # 获取监控状态
    status = monitor.get_monitor_status()
    print(f"监控状态: {status}")

