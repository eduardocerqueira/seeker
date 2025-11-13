#date: 2025-11-13T17:03:06Z
#url: https://api.github.com/gists/8dde3cf7f589cb5afb9a45bd262da41a
#owner: https://api.github.com/users/wangwei334455

"""
L2核心决策层 - 策略状态机（FSM）

职责：
1. 接收TICK数据，实时更新指标
2. 管理市场模式状态（震荡/上涨/下跌）
3. 执行策略决策，生成交易信号
4. 发送订单指令到L1
5. 接收L1订单反馈
6. 推送状态到L3监控层
"""
import json
import time
import hashlib
import redis
import numpy as np
from threading import Thread, Event, Lock
from typing import Optional, Dict, Any
from collections import deque
from loguru import logger

# 导入配置
try:
    from config.redis_config import REDIS_CONFIG, REDIS_KEYS
except ImportError:
    REDIS_CONFIG = {'host': 'localhost', 'port': 6379, 'db': 0, 'decode_responses': True}
    REDIS_KEYS = {}

# 导入核心模块
from src.trading.core.config_manager import ConfigManager
from src.trading.core.kline_builder import KlineBuilder, KLINE_DTYPE
from src.trading.core.kline_builder_enhanced import KlineBuilder as KlineBuilderEnhanced
from src.trading.core.indicators.micro_indicators import MicroContext, calculate_lrs_slope_jit
from src.trading.core.indicators.macro_indicators import (
    MacroContext, 
    calculate_atr_jit, 
    calculate_bbands_jit, 
    calculate_adx_jit,
    calculate_rsi_jit
)
from src.trading.strategies.base_strategy import BaseStrategy, Signal, MarketMode
from src.trading.strategies.ranging_strategy import RangingStrategy
from src.trading.strategies.uptrend_strategy import UptrendStrategy
from src.trading.strategies.downtrend_strategy import DowntrendStrategy
from src.trading.execution.order_executor import L2_ORDER_QUEUE, L1_FEEDBACK_QUEUE

# --- 策略状态机定义 ---
class StrategyState:
    """策略状态机状态定义"""
    IDLE = 'IDLE'           # 空闲
    WAIT_ENTRY = 'WAIT_ENTRY'  # 等待入场条件满足
    OPEN_LONG = 'OPEN_LONG'    # 持有多头仓位
    OPEN_SHORT = 'OPEN_SHORT'  # 持有空头仓位
    WAIT_CLOSE = 'WAIT_CLOSE'  # 等待平仓信号

# L3监控Stream
L3_MONITOR_STREAM = 'l3:monitor:status'


class L2StrategyCore:
    """
    L2核心决策层 - 策略状态机
    
    【关键设计】单线程事件循环架构：
    - 所有指标计算、模式识别和信号生成必须在单次循环中完成
    - 禁止耗时的阻塞操作（如网络IO、文件IO）
    - 确保决策的确定性和原子性
    - 所有计算都在内存中完成，无IPC开销
    
    性能目标：
    - L2内部处理延迟：< 3ms (P95)
    - 端到端延迟：< 50ms (P95，包含MT5 API延迟)
    """
    
    def __init__(self, symbol: str = "BTCUSDm"):
        """
        初始化L2核心决策层
        
        Args:
            symbol: 交易品种
        """
        self.symbol = symbol
        self.stop_event = Event()
        
        # Redis连接
        self.r = redis.Redis(
            host=REDIS_CONFIG.get('host', 'localhost'),
            port=REDIS_CONFIG.get('port', 6379),
            db=REDIS_CONFIG.get('db', 0),
            decode_responses=False  # 二进制模式，用于接收TICK数据
        )
        
        # 配置管理器
        self.config_manager = ConfigManager()
        
        # 加载配置参数
        kline_period = int(self.config_manager.get('GLOBAL', 'KLINE_PERIOD_MIN', 1))
        # 【数据层优化】支持2天M1 K线历史（2880根）
        history_size = max(int(self.config_manager.get('GLOBAL', 'HISTORY_CANDLES_N', 20)), 2880)
        lrs_period = int(self.config_manager.get('RANGING', 'LRS_TICKS_PERIOD', 20))
        density_period_ms = int(self.config_manager.get('RANGING', 'DENSITY_PERIOD_MS', 500))
        # 【数据层优化】TICK缓存扩展到3000个
        tick_cache_size = int(self.config_manager.get('GLOBAL', 'TICK_CACHE_SIZE', 3000))
        
        # 🔴 架构重构：移除K线构建器，K线由独立的Kline Service构建
        # K线数据从Redis读取，不再在策略服务中构建
        self.kline_builder = None  # 不再使用K线构建器
        
        # 指标上下文
        self.micro_context = MicroContext(lrs_period, density_period_ms, tick_cache_size=tick_cache_size)
        self.macro_context = MacroContext(history_size, self.config_manager.get_all('GLOBAL'))
        
        # 监控服务（实时监控系统健康和数据流）
        self.monitor_service = None
        try:
            # 🔴 修复：使用正确的导入路径
            from src.trading.services.monitor_service import MonitorService
            self.monitor_service = MonitorService(self.symbol)
            logger.info("监控服务已初始化（数据健康、系统性能和风险监控已启用）")
        except ImportError:
            # 监控服务是可选的，如果不存在则静默跳过
            logger.debug("监控服务模块不存在（可选功能，已跳过）")
        except Exception as e:
            logger.warning(f"监控服务初始化失败（监控功能未启用）: {e}")
        
        # 交易执行服务（可选，如果启用则进行订单-决策绑定和风控）
        self.trade_executor_service = None
        try:
            # 🔴 修复：使用正确的导入路径
            from src.trading.services.trade_executor_service import TradeExecutorService
            
            # 构建风控配置（从ConfigManager读取）
            risk_config = {
                'max_position_size': float(self.config_manager.get('GLOBAL', 'MAX_POSITION_SIZE', 1.0)),
                'max_daily_loss_usd': float(self.config_manager.get('GLOBAL', 'MAX_DAILY_LOSS_USD', 500.0)),
                'max_trades_per_minute': int(self.config_manager.get('GLOBAL', 'MAX_TRADES_PER_MINUTE', 10)),
                'max_order_risk_atr': float(self.config_manager.get('GLOBAL', 'MAX_ORDER_RISK_ATR', 2.0))
            }
            
            self.trade_executor_service = TradeExecutorService(REDIS_CONFIG, self.symbol, risk_config)
            
            # 将监控服务传递给交易执行服务（用于风险突破告警）
            if self.monitor_service:
                self.trade_executor_service.set_monitor_service(self.monitor_service)
            
            logger.info("交易执行服务已初始化（订单-决策绑定和实时风控已启用）")
        except Exception as e:
            logger.warning(f"交易执行服务初始化失败（订单-决策绑定和风控未启用）: {e}")
        
        # 市场模式状态
        self.current_mode = MarketMode.UNKNOWN
        self.current_strategy: Optional[BaseStrategy] = None
        
        # FSM状态锁（保护模式切换的原子性）
        self._fsm_lock = Lock()
        
        # 数据接收队列（从L1接收TICK数据）
        # 🔴 架构改进：从已验证的TICK流读取（数据完整性服务已验证）
        # 原始流: tick:{symbol}:stream
        # 已验证流: tick:{symbol}:validated:stream
        validated_stream_key = f'tick:{symbol}:validated:stream'
        self.tick_stream_key = validated_stream_key  # 使用已验证的流
        
        # TICK序列号追踪
        self.last_processed_seq = 0  # 最后处理的TICK序列号
        
        # ==================== DGTP策略：仓位管理系统 ====================
        # 仓位跟踪字典：{ 'BUY': [{ 'volume': 0.01, 'entry_price': 99.51, 'status': 'INITIAL', 'order_id': 123 }], 'SELL': [...] }
        self.dgtp_positions: Dict[str, list] = {'BUY': [], 'SELL': []}
        
        # ==================== DGTP策略：分钟线上下文（宏观判断）====================
        # 用于存储K线级别的市场区域判断
        self.kline_context = {
            'is_low_zone': False,      # 是否处于低点区域（分钟线判断）
            'is_high_zone': False,     # 是否处于高点区域（分钟线判断）
            'last_kline_time': 0,      # 最后更新的K线时间
        }
        
        # ==================== 宽幅震荡过滤机制 =====================
        # 用于过滤震荡噪音，避免频繁交易
        self.channel_filter = {
            'breakout_direction': None,  # 突破方向：'UP'（向上突破）、'DOWN'（向下突破）、None（通道内）
            'breakout_klines': 0,        # 突破后维持的K线数量
            'last_channel_check_time': 0,  # 最后检查通道的时间
        }
        
        # ==================== 微观动能刷单叠加模块（V2）=====================
        # 独立于主网格的高频套利层，用于快速刷单
        # 核心原则：固定仓位0.01手，只针对最后建仓的多单，独立结算
        self.scalping_positions: Dict[str, list] = {'BUY': [], 'SELL': []}  # 刷单仓位（独立管理，独立结算）
        self.scalping_pnl: Dict[str, float] = {'BUY': 0.0, 'SELL': 0.0}  # 刷单独立盈亏（独立于主网格PnL）
        self.scalping_state = {
            'last_scalp_time': 0,  # 最后一次刷单时间
            'waiting_reentry': None,  # 等待二次入场的方向：'BUY'或'SELL'或None
            'reentry_price': 0.0,  # 二次入场目标价格
            'reentry_retracement': 0.0,  # 回撤幅度（用于计算二次入场价格）
        }
        
        # ==================== 通用策略状态机（FSM）=====================
        # 用于基于宏观/微观指标融合的通用决策框架
        self.fsm_state = StrategyState.IDLE  # 当前FSM状态
        self.fsm_position_info = {
            'side': 'FLAT',      # 持仓方向：'LONG', 'SHORT', 'FLAT'
            'entry_price': 0.0,  # 入场价格
            'timestamp': 0,      # 入场时间戳
            'initial_atr': 0.0   # 入场时的ATR值（用于动态止损）
        }
        
        # 策略参数（可配置）
        self.ENTRY_LRS_THRESHOLD = float(self.config_manager.get('GLOBAL', 'ENTRY_LRS_THRESHOLD', 0.05))
        self.EXIT_RSI_THRESHOLD = float(self.config_manager.get('GLOBAL', 'EXIT_RSI_THRESHOLD', 70.0))
        self.RISK_ATR_MULTIPLIER = float(self.config_manager.get('GLOBAL', 'RISK_ATR_MULTIPLIER', 1.5))
        self.MAIN_TIMEFRAME = 'M1'  # 主要决策周期
        
        # 密度过滤参数（配置化）
        self.DENSITY_FILTER_ACTIVE = self.config_manager.get('GLOBAL', 'DENSITY_FILTER_ACTIVE', 'true').lower() == 'true'
        self.DENSITY_FILTER_TYPE = self.config_manager.get('GLOBAL', 'DENSITY_FILTER_TYPE', 'MOMENTUM_CONFIRM')
        # 支持的类型：
        # - 'MOMENTUM_CONFIRM': 动量突破策略，要求密度高于平均值（确认真实突破）
        # - 'ARBITRAGE': 套利策略，要求密度接近或低于平均值（稳定价差）
        # - 'REVERSAL': 反转策略，低密度时入场（等待价格回归）
        # - 'NONE': 不启用密度过滤
        self.DENSITY_AVG_MULTIPLIER = float(self.config_manager.get('GLOBAL', 'DENSITY_AVG_MULTIPLIER', 1.2))
        # 用于MOMENTUM_CONFIRM：密度需高于历史平均值的倍数
        self.DENSITY_ARBITRAGE_RANGE = float(self.config_manager.get('GLOBAL', 'DENSITY_ARBITRAGE_RANGE', 0.8))
        # 用于ARBITRAGE：密度应在平均值的此范围内（0.8表示80%-120%）
        
        # 决策ID生成器（用于订单-决策绑定）
        self.decision_counter = 0
        self.decision_lock = Lock()  # 保护决策计数器
        
        # 加载DGTP配置参数
        self._load_dgtp_config()
        
        # 4. 🔴 架构重构：从Redis加载历史K线数据（由Kline Service构建）
        self._load_historical_klines_from_redis()
        
        # 5. JIT预热（Warmup）
        logger.info("L2 Core: 开始JIT预热...")
        self._jit_warmup()
        logger.info("L2 Core: JIT预热完成")
        
        # 5. 启动监听线程
        self.feedback_thread = Thread(target=self._feedback_listener, daemon=True, name="L2FeedbackListener")
        self.feedback_thread.start()
        logger.info("L2 Core: 订单反馈监听线程已启动")
        
        # 数据接收线程
        self.data_receiver_thread = Thread(target=self._data_receiver_loop, daemon=True, name="L2DataReceiver")
        self.data_receiver_thread.start()
        logger.info("L2 Core: 数据接收线程已启动")
        
        # 最后信号
        self.last_signal = Signal.NONE
    
    def _load_dgtp_config(self):
        """加载DGTP策略配置参数"""
        mode_name = self.current_mode.name if self.current_mode != MarketMode.UNKNOWN else 'RANGING'
        
        self.initial_lot = float(self.config_manager.get(mode_name, 'INITIAL_LOT', 0.01))
        self.grid_step_atr = float(self.config_manager.get(mode_name, 'GRID_STEP_ATR', 1.5))
        self.max_ranging_avg = int(self.config_manager.get('RANGING', 'MAX_RANGING_AVG', 3))
        self.hedge_multiplier = float(self.config_manager.get('RANGING', 'HEDGE_MULTIPLIER', 2.0))
        self.deep_loss_atr = float(self.config_manager.get('RANGING', 'DEEP_LOSS_ATR', 2.0))
        self.profit_scalp_step_atr = float(self.config_manager.get('RANGING', 'PROFIT_SCALP_STEP_ATR', 1.5))
        self.max_pyramid_count = int(self.config_manager.get('UPTREND', 'MAX_PYRAMID_COUNT', 5))
        # 震荡翻转参数
        self.range_flip_multiple = float(self.config_manager.get('RANGING', 'RANGE_FLIP_MULTIPLE', 4.0))  # 震荡翻转的触发距离（例如4倍ATR）
        # 动量反应参数（硬编码"速度"因素，用于加速对冲，保持仓位纪律）
        self.momentum_threshold = float(self.config_manager.get('RANGING', 'MOMENTUM_THRESHOLD', 0.001))  # 动量阈值（价格变化速度超过此值视为急跌/急涨）
        self.momentum_compression_max = float(self.config_manager.get('RANGING', 'MOMENTUM_COMPRESSION_MAX', 0.9))  # 动量因子最大值（0.9表示步长压缩到10%）
        self.momentum_compression_min = float(self.config_manager.get('RANGING', 'MOMENTUM_COMPRESSION_MIN', 0.0))  # 动量因子最小值（0.0表示不压缩，使用标准步长）
        # 宽幅震荡过滤参数
        self.channel_atr_multiplier = float(self.config_manager.get('RANGING', 'CHANNEL_ATR_MULTIPLIER', 2.0))  # 通道ATR倍数（±2.0×ATR）
        self.breakout_confirmation_klines = int(self.config_manager.get('RANGING', 'BREAKOUT_CONFIRMATION_KLINES', 2))  # 突破确认所需的K线数量（默认2根）
        self.enable_channel_filter = bool(self.config_manager.get('RANGING', 'ENABLE_CHANNEL_FILTER', True))  # 是否启用通道过滤（默认启用）
        # 微观动能刷单参数（V2：固定仓位，只针对最后仓位）
        self.enable_scalping = bool(self.config_manager.get('RANGING', 'ENABLE_SCALPING', True))  # 是否启用刷单模块（默认启用）
        self.scalping_fixed_lot = float(self.config_manager.get('RANGING', 'SCALPING_FIXED_LOT', 0.01))  # 刷单固定仓位（0.01手，不随主网格变化）
        self.momentum_entry_threshold = float(self.config_manager.get('RANGING', 'MOMENTUM_ENTRY_THRESHOLD', 1.5))  # 动量入场阈值（ΔP > 1.5）
        self.decay_exit_threshold = float(self.config_manager.get('RANGING', 'DECAY_EXIT_THRESHOLD', 0.7))  # 动能衰竭平仓阈值（Decay > 0.7）
        self.scalping_sl_points = float(self.config_manager.get('RANGING', 'SCALPING_SL_POINTS', 0.0008))  # 刷单固定止损（5-8个基点，例如0.0008）
        self.reentry_retracement_ratio = float(self.config_manager.get('RANGING', 'REENTRY_RETRACEMENT_RATIO', 0.382))  # 二次入场回撤比例（斐波那契0.382）
    
    def _jit_warmup(self):
        """
        JIT预热：通过调用JIT函数模拟简单数据，触发Numba编译
        
        这确保L2启动后能立即进入低延迟状态，避免首次调用的编译延迟
        """
        try:
            # 预热Micro Indicators
            test_prices = np.array([100.0, 100.1, 100.2, 100.3, 100.4], dtype=np.float64)
            _ = calculate_lrs_slope_jit(test_prices)
            logger.debug("L2 Core: LRS JIT预热完成")
            
            # 预热Macro Indicators
            N = self.macro_context.history_size
            if N < 20:
                N = 20  # 确保至少20个数据点
            
            # 生成测试数据
            np.random.seed(42)  # 固定随机种子，确保可重复
            test_high = np.random.rand(N).astype(np.float64) * 10 + 100
            test_low = test_high - np.random.rand(N).astype(np.float64) * 2
            test_close = (test_high + test_low) / 2
            period = max(N // 2, 10)
            
            # 预热ATR, BBANDS, ADX
            _ = calculate_atr_jit(test_high, test_low, test_close, period)
            logger.debug("L2 Core: ATR JIT预热完成")
            
            _ = calculate_bbands_jit(test_close, period, 2.0)
            logger.debug("L2 Core: BBANDS JIT预热完成")
            
            _ = calculate_adx_jit(test_high, test_low, test_close, period)
            logger.debug("L2 Core: ADX JIT预热完成")
            
        except Exception as e:
            logger.warning(f"L2 Core: JIT预热过程中出现错误（不影响运行）: {e}")
    
    def _read_latest_closed_kline_from_redis(self) -> Optional[Dict[str, Any]]:
        """
        从Redis读取最新闭合的K线（用于宏观指标计算）
        
        Returns:
            最新闭合的K线字典，如果没有则返回None
        """
        try:
            r_text = redis.Redis(
                host=REDIS_CONFIG.get('host', 'localhost'),
                port=REDIS_CONFIG.get('port', 6379),
                db=REDIS_CONFIG.get('db', 0),
                decode_responses=True
            )
            
            kline_key = f"kline:{self.symbol}:1m"
            # 获取最新的K线（最后一条）
            klines_json = r_text.zrange(kline_key, -1, -1)
            
            if klines_json:
                kline = json.loads(klines_json[0])
                return kline
            
            return None
        except Exception as e:
            logger.debug(f"L2 Core: 读取最新K线失败（非关键）: {e}")
            return None
    
    def _load_historical_klines_from_redis(self):
        """
        🔴 架构重构：从Redis加载历史K线数据（由Kline Service构建）
        
        不再从MT5或本地构建K线，而是从Redis读取已构建的K线数据
        用于初始化MacroContext
        """
        try:
            # 从Redis Sorted Set读取最近2880根K线（2天M1数据）
            kline_key = f"kline:{self.symbol}:1m"
            klines_data = self.r.zrange(kline_key, -2880, -1, withscores=False)
            
            if not klines_data:
                logger.warning(f"L2 Core: Redis中没有找到历史K线数据（key: {kline_key}）")
                return
            
            # 解析JSON数据
            klines = []
            for kline_json in klines_data:
                try:
                    kline = json.loads(kline_json)
                    klines.append({
                        'time': int(kline['time']),
                        'open': float(kline['open']),
                        'high': float(kline['high']),
                        'low': float(kline['low']),
                        'close': float(kline['close']),
                        'volume': int(kline.get('volume', 0))
                    })
                except Exception as e:
                    logger.warning(f"L2 Core: 解析K线数据失败: {e}")
                    continue
            
            if klines:
                self.macro_context.load_historical_klines(klines)
                logger.info(f"L2 Core: 成功加载 {len(klines)} 根历史K线数据")
            else:
                logger.warning("L2 Core: 没有有效的历史K线数据")
                
        except Exception as e:
            logger.error(f"L2 Core: 加载历史K线数据失败: {e}")
    
    def _data_receiver_loop(self):
        """
        数据接收循环：从Redis Stream接收TICK数据
        
        【单线程事件循环】这是L2的核心循环，负责：
        1. 接收TICK数据（阻塞读取，100ms超时）
        2. 更新指标上下文（微观/宏观）
        3. 执行策略决策（模式识别、信号生成）
        4. 发送订单指令（推送到Redis List）
        5. 推送状态到L3（异步，不阻塞）
        
        【性能要求】
        - 所有计算必须在单次循环中完成，禁止阻塞操作
        - 确保决策的确定性和原子性
        - 目标延迟：< 3ms (L2内部处理)
        
        增强错误处理：Redis连接重试和异常恢复
        """
        logger.info("L2 Core: 数据接收循环已启动")
        logger.info(f"L2 Core: 监听已验证流 - {self.tick_stream_key}")
        
        # 🔴 修复：从流的末尾开始读取（只处理新数据）
        # 使用 '$' 表示只读取新消息，避免处理历史数据
        last_id = '$'
        reconnect_delay = 1.0
        
        while not self.stop_event.is_set():
            try:
                # 确保Redis连接存在（可能被内部连接池管理）
                try:
                    self.r.ping()
                except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError):
                    # 连接丢失，尝试重新创建
                    logger.warning(f"L2 Receiver: Redis连接丢失，{reconnect_delay}秒后重试...")
                    time.sleep(reconnect_delay)
                    reconnect_delay = min(reconnect_delay * 2, 10.0)  # 指数退避，最大10秒
                    
                    try:
                        self.r = redis.Redis(
                            host=REDIS_CONFIG.get('host', 'localhost'),
                            port=REDIS_CONFIG.get('port', 6379),
                            db=REDIS_CONFIG.get('db', 0),
                            decode_responses=False
                        )
                        self.r.ping()
                        reconnect_delay = 1.0  # 重置延迟
                        logger.info("L2 Receiver: Redis连接已恢复")
                    except Exception as reconnect_error:
                        logger.error(f"L2 Receiver: Redis重连失败: {reconnect_error}")
                        continue
                
                # 从Redis Stream读取TICK数据（阻塞读取，超时100ms）
                messages = self.r.xread({self.tick_stream_key: last_id}, count=1, block=100)
                
                if not messages:
                    continue
                
                # 处理消息
                for stream, msgs in messages:
                    for msg_id, msg_data in msgs:
                        # 解析TICK数据（二进制格式）
                        tick_data = self._parse_tick_data(msg_data)
                        
                        if tick_data:
                            # 处理TICK数据
                            self._handle_tick(tick_data)
                        
                        # 更新最后处理的ID
                        last_id = msg_id
                
            except redis.exceptions.ConnectionError as ce:
                logger.warning(f"L2 Receiver: Redis连接错误: {ce}")
                time.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, 10.0)
            except redis.exceptions.TimeoutError as te:
                logger.warning(f"L2 Receiver: Redis超时: {te}")
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"L2 Core: 数据接收意外错误: {e}")
                time.sleep(0.01)  # 短暂等待，避免CPU空转
        
        logger.info("L2 Core: 数据接收循环已停止")
    
    def _parse_tick_data(self, msg_data: Dict[bytes, bytes]) -> Optional[Dict[str, Any]]:
        """
        解析TICK数据（从Redis Stream二进制格式）
        
        【数据格式】
        - Data Integrity Service 写入格式：{'value': json_string}
        - 需要解析 value 字段中的 JSON 字符串
        
        Args:
            msg_data: Redis Stream消息数据
            
        Returns:
            TICK数据字典
        """
        try:
            # 🔴 修复：Data Integrity Service 使用 'value' 字段存储 JSON
            if b'value' in msg_data:
                value_json = msg_data[b'value'].decode('utf-8')
                return json.loads(value_json)
            # 兼容旧格式：{'data': json_string}
            elif b'data' in msg_data:
                data_json = msg_data[b'data'].decode('utf-8')
                return json.loads(data_json)
            else:
                # 尝试直接解析所有字段（如果已经是字典格式）
                tick_data = {}
                for key, value in msg_data.items():
                    key_str = key.decode('utf-8')
                    value_str = value.decode('utf-8')
                    # 如果 value 是 JSON 字符串，尝试解析
                    if key_str == 'value' or key_str == 'data':
                        try:
                            return json.loads(value_str)
                        except:
                            pass
                    tick_data[key_str] = value_str
                return tick_data
        except Exception as e:
            logger.error(f"L2 Core: 解析TICK数据失败: {e}")
            return None
    
    def _handle_tick(self, tick_data: Dict[str, Any]):
        """
        处理TICK数据（核心决策逻辑）
        
        【数据完整性验证】
        1. Seq顺序检查：确保TICK按顺序接收
        2. Checksum验证：确保数据未被篡改
        
        Args:
            tick_data: TICK数据字典（包含seq和checksum）
        """
        try:
            # --- A. 数据完整性检查 ---
            
            # 🔴 架构改进：数据完整性检查已由Data Integrity Service完成
            # 策略核心不再负责数据验证（seq、checksum），只记录seq用于监控
            current_seq = tick_data.get('seq', 0)
            if current_seq > 0:
                # 仅用于监控和日志，不进行验证（已验证流中的数据已经验证过）
                if current_seq != self.last_processed_seq + 1:
                    logger.debug(f"L2 Core: Seq={current_seq}（已验证流，仅记录）")
                self.last_processed_seq = current_seq
            
            # --- B. 核心数据更新与分发 ---
            
            # 记录FSM开始处理时间（用于性能监控）
            fsm_start_time = time.time()
            
            # 1. 提取TICK信息
            time_msc = int(tick_data.get('time_msc', time.time() * 1000))
            # 🔴 修复：如果last为0，使用(bid+ask)/2作为价格
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
                logger.warning(f"L2 Core: TICK价格无效，跳过处理 (Seq: {tick_data.get('seq')})")
                return
            
            volume = float(tick_data.get('volume', 0.0))
            
            # 1.1. 报告TICK健康状态（监控数据延迟和心跳）
            if self.monitor_service:
                self.monitor_service.report_tick_health(tick_data, fsm_start_time)
            
            # 2. 更新微观指标上下文
            self.micro_context.update_context(time_msc, price)
            
            # 3. 🔴 架构重构：不再构建K线，改为从Redis读取最新K线
            # K线由独立的Kline Service构建和存储
            # 策略服务只负责读取K线数据用于指标计算
            
            # 从Redis读取最新闭合的K线（用于宏观指标计算）
            # 🔴 架构重构：K线由Kline Service构建，策略服务只读取
            latest_closed_kline = self._read_latest_closed_kline_from_redis()
            if latest_closed_kline:
                # 检查是否是新K线（通过比较时间戳）
                # 如果K线历史为空或新K线时间大于最后K线时间，则更新
                if len(self.macro_context.kline_history) == 0:
                    # 首次加载，直接更新
                    kline_array = np.array([(
                        latest_closed_kline['time'],
                        latest_closed_kline['open'],
                        latest_closed_kline['high'],
                        latest_closed_kline['low'],
                        latest_closed_kline['close'],
                        latest_closed_kline.get('volume', 0)
                    )], dtype=KLINE_DTYPE)
                    self.macro_context.update_context(kline_array)
                else:
                    # 检查是否是新K线
                    last_kline = self.macro_context.kline_history[-1]
                    last_kline_time = last_kline['time'][0] if isinstance(last_kline, np.ndarray) else last_kline['time']
                    if latest_closed_kline.get('time', 0) > last_kline_time:
                        kline_array = np.array([(
                            latest_closed_kline['time'],
                            latest_closed_kline['open'],
                            latest_closed_kline['high'],
                            latest_closed_kline['low'],
                            latest_closed_kline['close'],
                            latest_closed_kline.get('volume', 0)
                        )], dtype=KLINE_DTYPE)
                        self.macro_context.update_context(kline_array)
            
            # 5. 延迟监控
            delay = time.time() - (time_msc / 1000.0)
            if delay > 0.5:  # 警告：处理延迟超过500ms
                logger.warning(f"L2 Core: TICK处理延迟: {delay:.3f}s (Seq: {current_seq})")
            
            # 6. 更新状态
            self.last_processed_seq = current_seq
            # 检查模式切换
            self._check_mode_switch()
            # 更新分钟线上下文（判断低点/高点区域）
            self._update_kline_context(price)
            # 更新通道过滤状态（检测突破和时间确认）
            self._update_channel_filter(price)
            
            # 7. 执行通用策略决策（基于宏观/微观指标融合的FSM）
            self._make_decision(tick_data)
            
            # 6. 执行DGTP策略决策（TICK级别精确进出场）
            self._execute_dgtp_strategy(price)
            
            # 7. 执行微观动能刷单模块（独立于主网格，仅在K线收盘时执行）
            # 🔴 修复：使用从Redis读取的最新闭合K线，转换为numpy array格式
            if self.enable_scalping and latest_closed_kline is not None:
                try:
                    # 将字典转换为numpy structured array格式（_execute_scalping_overlay期望此格式）
                    closed_kline_array = np.array([(
                        latest_closed_kline['time'],
                        latest_closed_kline['open'],
                        latest_closed_kline['high'],
                        latest_closed_kline['low'],
                        latest_closed_kline['close'],
                        latest_closed_kline.get('volume', 0)
                    )], dtype=KLINE_DTYPE)
                    # 验证数组格式正确
                    if len(closed_kline_array) > 0:
                        self._execute_scalping_overlay(price, closed_kline_array)
                except (KeyError, TypeError, ValueError) as e:
                    logger.warning(f"L2 Core: 转换K线数据失败，跳过刷单模块: {e}")
                except Exception as e:
                    logger.error(f"L2 Core: 执行刷单模块时出错: {e}")
                    import traceback
                    logger.debug(f"L2 Core: 刷单模块错误堆栈: {traceback.format_exc()}")
            
            # 8. 推送状态到L3监控层
            self._push_status_to_l3(time_msc)
            
            # 9. 报告FSM性能（监控处理耗时）
            if self.monitor_service:
                try:
                    fsm_end_time = time.time()
                    loop_duration_ms = (fsm_end_time - fsm_start_time) * 1000.0
                    self.monitor_service.report_fsm_performance(loop_duration_ms)
                except Exception as monitor_error:
                    # 监控服务错误不应影响主流程
                    logger.debug(f"L2 Core: 监控服务报告失败: {monitor_error}")
            
        except Exception as e:
            import traceback
            # 改进错误信息显示
            try:
                error_msg = str(e) if e else repr(e)
                error_type = type(e).__name__
                # 如果错误信息是"0"，尝试获取更详细的信息
                if error_msg == "0" or error_msg == 0:
                    error_msg = f"{repr(e)} (type: {type(e)}, args: {e.args if hasattr(e, 'args') else 'N/A'})"
                logger.error(f"L2 Core: 处理TICK数据错误 [{error_type}]: {error_msg}")
                logger.debug(f"L2 Core: 错误堆栈: {traceback.format_exc()}")
            except Exception as log_error:
                # 如果连日志都失败了，使用最基本的输出
                logger.error(f"L2 Core: 处理TICK数据错误（日志记录失败）: {e}, {log_error}")
    
    def _check_mode_switch(self):
        """
        检查市场模式是否需要切换（K线级FSM守护）
        
        基于ADX和价格位置判断趋势确立/终结
        """
        # 获取宏观指标
        adx = self.macro_context.get_adx()
        bb_upper, bb_mid, bb_lower = self.macro_context.get_bbands()
        
        # 获取当前价格
        current_price = 0.0
        if self.micro_context.tick_buffer:
            # tick_buffer存储的是(time_msc, price)元组
            current_price = self.micro_context.tick_buffer[-1][1]
        
        if current_price == 0.0 or bb_mid == 0.0:
            return
        
        # 获取配置阈值
        adx_min_trend = float(self.config_manager.get('UPTREND', 'ADX_MIN_THRESHOLD', 30.0))
        adx_max_ranging = float(self.config_manager.get('RANGING', 'ADX_MAX_THRESHOLD', 25.0))
        
        # 趋势强度判断
        if adx > adx_min_trend:
            # 趋势确立：切换到趋势模式
            if current_price > bb_mid:
                # 价格在中轨之上，判断为上涨
                new_mode = MarketMode.UPTREND
            elif current_price < bb_mid:
                # 价格在中轨之下，判断为下跌
                new_mode = MarketMode.DOWNTREND
            else:
                # 价格在中轨附近，保持当前模式
                return
        elif adx < adx_max_ranging:
            # 趋势衰减：切换回震荡模式
            new_mode = MarketMode.RANGING
        else:
            # ADX在中间值，保持当前模式
            return
        
        # 如果模式改变，切换策略
        if new_mode != self.current_mode:
            self._switch_mode(new_mode)
    
    # ==================== DGTP策略：分钟线上下文更新 ====================
    
    def _update_kline_context(self, current_price: float):
        """
        更新分钟线上下文（宏观判断）
        
        在K线收盘时调用，判断当前处于低点区域还是高点区域
        这是"低多高平"的宏观基础
        
        Args:
            current_price: 当前价格
        """
        # 获取布林带
        bb_upper, bb_mid, bb_lower = self.macro_context.get_bbands()
        if bb_upper == 0.0 or bb_lower == 0.0:
            return
        
        atr = self.macro_context.get_atr()
        if atr == 0.0:
            return
        
        step = self.grid_step_atr * atr
        
        # 判断价格区域（分钟线级别）
        # 低点区域：价格在下轨附近（下轨 + step/2 范围内）
        # 高点区域：价格在上轨附近（上轨 - step/2 范围内）
        self.kline_context['is_low_zone'] = current_price <= (bb_lower + step / 2)
        self.kline_context['is_high_zone'] = current_price >= (bb_upper - step / 2)
        
        # 🔴 修复：移除频繁的日志输出，只在上下文状态变化时记录（降低日志噪音）
        # logger.debug(f"L2 DGTP: 分钟线上下文更新 - 低点区域: {self.kline_context['is_low_zone']}, "
        #             f"高点区域: {self.kline_context['is_high_zone']}, 价格: {current_price:.4f}")
    
    def _update_channel_filter(self, current_price: float):
        """
        更新通道过滤状态（检测突破和时间确认）
        
        用于过滤震荡噪音，避免频繁交易
        
        Args:
            current_price: 当前价格
        """
        # 获取布林带
        bb_upper, bb_mid, bb_lower = self.macro_context.get_bbands()
        if bb_upper == 0.0 or bb_lower == 0.0:
            return
        
        # 检测突破方向
        if current_price > bb_upper:
            # 向上突破
            if self.channel_filter['breakout_direction'] != 'UP':
                self.channel_filter['breakout_direction'] = 'UP'
                self.channel_filter['breakout_klines'] = 0
            else:
                self.channel_filter['breakout_klines'] += 1
        elif current_price < bb_lower:
            # 向下突破
            if self.channel_filter['breakout_direction'] != 'DOWN':
                self.channel_filter['breakout_direction'] = 'DOWN'
                self.channel_filter['breakout_klines'] = 0
            else:
                self.channel_filter['breakout_klines'] += 1
        else:
            # 在通道内，重置突破状态
            if self.channel_filter['breakout_direction'] is not None:
                self.channel_filter['breakout_direction'] = None
                self.channel_filter['breakout_klines'] = 0
        
        # 更新最后检查时间
        self.channel_filter['last_channel_check_time'] = int(time.time())
    
    def _is_in_channel(self, price: float) -> bool:
        """
        检查价格是否在布林带通道内（震荡噪音区）
        
        Args:
            price: 当前价格
            
        Returns:
            True: 价格在通道内（bb_lower <= price <= bb_upper）
            False: 价格在通道外（已突破）
        """
        # 获取布林带
        bb_upper, bb_mid, bb_lower = self.macro_context.get_bbands()
        if bb_upper == 0.0 or bb_lower == 0.0:
            # 布林带未初始化，默认返回False（不在通道内）
            return False
        
        # 价格在通道内：bb_lower <= price <= bb_upper
        return bb_lower <= price <= bb_upper
    
    def _is_channel_breakout_confirmed(self, direction: str) -> bool:
        """
        检查通道突破是否已确认（需要连续N根K线在通道外）
        
        Args:
            direction: 突破方向 ('UP' 或 'DOWN')
            
        Returns:
            True: 突破已确认
            False: 突破未确认
        """
        # 使用已加载的配置值（与初始化时保持一致）
        return (self.channel_filter['breakout_direction'] == direction and
                self.channel_filter['breakout_klines'] >= self.breakout_confirmation_klines)
    
    # ==================== DGTP策略：仓位管理辅助方法 ====================
    
    def _get_total_volume(self, side: str) -> float:
        """计算某一方向的总仓位"""
        return sum(pos.get('volume', 0.0) for pos in self.dgtp_positions.get(side, []))
    
    def _get_anchor_price(self, side: str) -> Optional[float]:
        """
        获取某一方向的锚定价格（最差的入场价）
        
        用于确保加仓逻辑的安全性和步长的一致性
        """
        positions = self.dgtp_positions.get(side, [])
        if not positions:
            return None
        
        # 使用最差的（最不盈利的）入场价作为锚点
        if side == 'BUY':
            return min(pos.get('entry_price', float('inf')) for pos in positions)
        else:  # SELL
            return max(pos.get('entry_price', 0.0) for pos in positions)
    
    def _get_anchor_position(self, side: str) -> Optional[Dict[str, Any]]:
        """
        获取某一方向的 ANCHOR 仓位
        
        Args:
            side: 方向（'BUY' 或 'SELL'）
            
        Returns:
            ANCHOR 仓位字典，如果没有则返回 None
        """
        return next((pos for pos in self.dgtp_positions.get(side, []) if pos.get('type') == 'ANCHOR'), None)
    
    def _close_all_pyramided_positions(self):
        """
        平掉所有加仓仓位（PYRAMID），只保留锚定仓位（ANCHOR）和对冲仓位（HEDGE）
        """
        for side in ['BUY', 'SELL']:
            pyramid_positions = [pos for pos in self.dgtp_positions.get(side, []) if pos.get('type') == 'PYRAMID']
            if pyramid_positions:
                close_volume = sum(pos['volume'] for pos in pyramid_positions)
                self._send_dgtp_order('CLOSE_POSITION_BY_TYPE', side, close_volume, f'FSM_RANGE_RESET_{side}')
                
                # 只保留锚定仓位和对冲仓位
                self.dgtp_positions[side] = [pos for pos in self.dgtp_positions[side] 
                                            if pos.get('type') in ['ANCHOR', 'HEDGE']]
                logger.info(f"L2 DGTP: RANGE RESET - 平掉 {close_volume} 的 {side} 加仓仓位，保留锚定仓位")
    
    def _publish_signal_to_l3_executor(self, command: Dict[str, Any]):
        """
        L2向L3 Tick执行上下文发布交易意图
        
        职责分离：
        - L2 (分钟线决策层): 基于ATR、BBands、ADX等分钟级指标判断市场状态，生成交易意图（Intent）
        - L3 (TICK执行层): 接收意图后，立即监控TICK数据，使用VWAP/TWAP/Iceberg等算法优化成交价格
        
        Args:
            command: 交易意图命令字典
        """
        # 推送到Redis List（L3执行层监听此队列）
        try:
            self.r.lpush(L2_ORDER_QUEUE, json.dumps(command))
            logger.info(f"L2 DGTP: [分钟线决策] 发布交易意图到L3 - {command.get('action')} {command.get('type')}, "
                       f"volume={command.get('volume')}, comment={command.get('comment')}")
        except Exception as e:
            logger.error(f"L2 DGTP: 发布交易意图失败: {e}")
    
    def _send_dgtp_order(self, action: str, side: str, volume: float, comment: str):
        """
        L2构建交易意图并发送到L3 Tick执行层
        
        职责：
        - L2 (分钟线决策层): 基于ATR、BBands等分钟级指标生成交易意图（Intent）
        - L3 (TICK执行层): 接收意图后，使用TICK数据优化成交价格（VWAP、TWAP、Iceberg等）
        
        Args:
            action: 订单动作（'PLACE_ORDER', 'CLOSE_POSITION_BY_TYPE', 'CLOSE_ALL_POSITIONS_BY_TYPE'）
            side: 方向（'BUY' 或 'SELL'）
            volume: 交易量
            comment: 订单备注（包含决策上下文信息）
        """
        # 获取当前价格（用于参考，实际执行由L3优化）
        current_price = 0.0
        if self.micro_context.tick_buffer:
            # tick_buffer存储的是(time_msc, price)元组
            current_price = self.micro_context.tick_buffer[-1][1]
        
        if current_price == 0.0:
            logger.warning("L2 DGTP: 无法获取当前价格，跳过订单意图")
            return
        
        # 构建交易意图指令
        if action == 'PLACE_ORDER':
            order_action = 'BUY' if side == 'BUY' else 'SELL'
        elif action == 'CLOSE_POSITION_BY_TYPE':
            # 🔴 修复：OrderExecutor只支持CLOSE_ALL，不支持CLOSE_POSITION
            # 对于按类型平仓，我们使用CLOSE_ALL来平掉指定方向的所有仓位
            order_action = 'CLOSE_ALL'
        elif action == 'CLOSE_ALL_POSITIONS_BY_TYPE':
            order_action = 'CLOSE_ALL'
        else:
            logger.warning(f"L2 DGTP: 未知订单动作: {action}")
            return
        
        # 获取ATR计算止损（分钟线级别风控）
        atr = self.macro_context.get_atr()
        atr_sl_mult = float(self.config_manager.get(self.current_mode.name, 'ATR_SL_MULTIPLIER', 2.0))
        
        # 计算止损价格（L2决策层设定）
        if order_action in ('BUY', 'SELL'):
            if order_action == 'BUY':
                sl_price = current_price - atr * atr_sl_mult if atr > 0 else current_price * 0.99
            else:
                sl_price = current_price + atr * atr_sl_mult if atr > 0 else current_price * 1.01
        else:
            sl_price = 0.0
        
        # 计算止盈价格（如果配置了止盈逻辑）
        tp_price = self._calculate_take_profit(side, current_price, atr)
        
        # 🔴 修复：验证止损/止盈价格合理性，避免MT5返回"Invalid stops"错误
        # MT5要求：止损和止盈价格必须与当前价格有合理距离（至少10个点）
        min_distance = 10.0  # 最小距离（点）
        if order_action == 'BUY' and sl_price > 0:
            # 做多：止损必须低于当前价格，止盈必须高于当前价格
            if sl_price >= current_price - min_distance:
                # 止损太接近当前价格，调整为更合理的值
                sl_price = current_price - max(atr * atr_sl_mult, min_distance) if atr > 0 else current_price - min_distance
            if tp_price > 0 and tp_price <= current_price + min_distance:
                # 止盈太接近当前价格，不设置止盈
                tp_price = 0.0
            if tp_price > 0 and tp_price <= sl_price:
                # 止盈必须大于止损，否则不设置止盈
                tp_price = 0.0
        elif order_action == 'SELL' and sl_price > 0:
            # 做空：止损必须高于当前价格，止盈必须低于当前价格
            if sl_price <= current_price + min_distance:
                # 止损太接近当前价格，调整为更合理的值
                sl_price = current_price + max(atr * atr_sl_mult, min_distance) if atr > 0 else current_price + min_distance
            if tp_price > 0 and tp_price >= current_price - min_distance:
                # 止盈太接近当前价格，不设置止盈
                tp_price = 0.0
            if tp_price > 0 and tp_price >= sl_price:
                # 止盈必须小于止损，否则不设置止盈
                tp_price = 0.0
        
        # 构建交易意图命令
        # 注意：order_type='MARKET' 是默认值，但L3执行层会根据TICK数据优化为Limit Order或使用VWAP算法
        command = {
            'action': order_action,
            'symbol': self.symbol,
            'type': side,
            'volume': volume,
            'price': current_price,  # 参考价格，L3会优化
            'sl': round(sl_price, 5),  # 止损价格（L2决策）
            'tp': round(tp_price, 5) if tp_price > 0 else 0.0,  # 止盈价格（L2决策）
            'order_type': 'INTENT',  # 标记为意图，L3执行层负责优化
            'comment': comment,
            'execution_hint': 'TICK_OPTIMIZED'  # 提示L3使用TICK数据优化执行
        }
        
        # 发布交易意图到L3 Tick执行层
        self._publish_signal_to_l3_executor(command)
    
    def _calculate_take_profit_exit(self, price: float, side: str, atr: float) -> float:
        """
        实现复杂的群体止盈/回撤清仓逻辑（L2分钟线决策层）
        
        规则：倒数第N个加仓仓位（PYRAMID），若回退M个网格，则清掉所有 PYRAMID 仓位，只保留 ANCHOR 仓位。
        
        这是多层止损机制，用于在价格大幅回撤时保护利润，同时保留锚定仓位以捕捉后续反弹。
        
        Args:
            price: 当前价格
            side: 方向（'BUY' 或 'SELL'）
            atr: ATR值
            
        Returns:
            清仓的交易量（如果触发），否则返回0
        """
        active_positions = self.dgtp_positions.get(side, [])
        pyramid_positions = [pos for pos in active_positions if pos.get('type') == 'PYRAMID']
        
        # 至少需要 N 个 PYRAMID 仓位才能启动群体止损机制
        if len(pyramid_positions) < self.group_sl_n:
            return 0  # 不触发
        
        # 1. 确定群体止损的锚点仓位（倒数第N个 PYRAMID 仓位）
        # 注意：列表索引 [-(N)] 是倒数第N个元素
        sl_anchor_pos = pyramid_positions[-self.group_sl_n]
        sl_anchor_price = sl_anchor_pos['entry_price']
        
        # 2. 计算群体止损触发距离
        grid_step_distance = self.grid_step_atr * atr
        
        should_close_pyramids = False
        
        if side == 'BUY':
            # 做多时，价格回撤到 Nth_Entry - M * Step 时触发
            trigger_price = sl_anchor_price - (self.group_sl_steps * grid_step_distance)
            if price < trigger_price:
                should_close_pyramids = True
        elif side == 'SELL':
            # 做空时，价格回撤到 Nth_Entry + M * Step 时触发
            trigger_price = sl_anchor_price + (self.group_sl_steps * grid_step_distance)
            if price > trigger_price:
                should_close_pyramids = True
        
        if should_close_pyramids:
            # 3. 触发清仓所有 PYRAMID 仓位
            close_volume = sum(pos['volume'] for pos in pyramid_positions)
            
            if close_volume > 0:
                command = {
                    'action': 'CLOSE_POSITION_BY_TYPE',
                    'symbol': self.symbol,
                    'type': side,
                    'volume': close_volume,
                    'order_type': 'INTENT',
                    'comment': f'GROUP_SL_TRIGGER_{side}',
                    'execution_hint': 'TICK_OPTIMIZED'
                }
                self._publish_signal_to_l3_executor(command)
                
                # 4. 更新本地仓位（只保留 ANCHOR 仓位）
                self.dgtp_positions[side] = [pos for pos in active_positions if pos.get('type') == 'ANCHOR']
                logger.warning(f"L2 DGTP: [群体止损] 清仓 {close_volume} 的 {side} PYRAMID仓位，保留ANCHOR仓位")
                return close_volume
        
        return 0
    
    def _calculate_take_profit(self, side: str, current_price: float, atr: float) -> float:
        """
        计算止盈价格（L2分钟线决策层）
        
        止盈策略选项：
        1. 基于ATR倍数：TP = Entry ± ATR × TP_MULTIPLIER
        2. 基于BBands边界：TP = BB上轨（做多）或 BB下轨（做空）
        3. 基于网格步长：TP = Entry ± GRID_STEP_ATR × ATR
        
        Args:
            side: 方向（'BUY' 或 'SELL'）
            current_price: 当前价格
            atr: ATR值
            
        Returns:
            止盈价格（0.0表示不设置止盈）
        """
        # 获取止盈配置参数
        tp_strategy = self.config_manager.get('GLOBAL', 'TP_STRATEGY', 'ATR')  # 'ATR', 'BBANDS', 'GRID_STEP'
        tp_atr_mult = float(self.config_manager.get(self.current_mode.name, 'TP_ATR_MULTIPLIER', 0.0))
        
        if tp_strategy == 'ATR' and tp_atr_mult > 0:
            # 策略1：基于ATR倍数
            if side == 'BUY':
                return current_price + atr * tp_atr_mult
            else:  # SELL
                return current_price - atr * tp_atr_mult
        elif tp_strategy == 'BBANDS':
            # 策略2：基于BBands边界
            bb_upper, bb_mid, bb_lower = self.macro_context.get_bbands()
            if side == 'BUY':
                return bb_upper if bb_upper > 0 else 0.0
            else:  # SELL
                return bb_lower if bb_lower > 0 else 0.0
        elif tp_strategy == 'GRID_STEP':
            # 策略3：基于网格步长
            step = self.grid_step_atr * atr
            if side == 'BUY':
                return current_price + step
            else:  # SELL
                return current_price - step
        
        # 默认：不设置止盈（由L3执行层或手动管理）
        return 0.0
    
    # ==================== DGTP策略：震荡模式逻辑 ====================
    
    def _execute_ranging_dgtp(self, price: float):
        """
        震荡模式DGTP逻辑 - L2分钟线决策层
        
        核心原则（两层上下文分离）：
        - **L2分钟线上下文（宏观决策）**：基于BBands、ATR等分钟级指标判断"低点区域"和"高点区域"
        - **L3 TICK上下文（微观执行）**：接收L2意图后，使用TICK数据（LRS、TICK密度）优化成交价格
        
        策略流程：
        1. **低多**：L2判断价格处于低点区域 → 发送入场意图 → L3使用TICK数据优化进场价格
        2. **高平**：L2判断价格处于高点区域 + 盈利达到步长 → 发送出场意图 → L3使用TICK数据优化出场价格
        3. **高空**：L2判断价格处于高点区域 → 发送入场意图 → L3使用TICK数据优化进场价格
        4. **低平**：L2判断价格处于低点区域 + 盈利达到步长 → 发送出场意图 → L3使用TICK数据优化出场价格
        """
        atr = self.macro_context.get_atr()
        if atr == 0.0:
            return
        
        step = self.grid_step_atr * atr
        deep_loss_threshold = self.deep_loss_atr * atr
        
        # 获取微观指标（TICK级别）
        lrs = self.micro_context.get_lrs()
        density = self.micro_context.get_density()
        lrs_reverse_threshold = float(self.config_manager.get('RANGING', 'LRS_REVERSE_THRESHOLD', 0.00005))
        
        current_buy_vol = self._get_total_volume('BUY')
        current_sell_vol = self._get_total_volume('SELL')
        
        # 获取锚定价格（最差入场价，用于风控步长计算）
        buy_anchor = self._get_anchor_price('BUY')
        sell_anchor = self._get_anchor_price('SELL')
        
        # 获取分钟线上下文（宏观判断）
        is_low_zone = self.kline_context.get('is_low_zone', False)
        is_high_zone = self.kline_context.get('is_high_zone', False)
        
        # ** 1. 震荡翻转逻辑（最高优先级，但需要双重确认）**
        # 双重确认机制：避免在真突破时过早翻转
        # 机制一：价格达到预设区间边缘
        # 机制二：指标确认市场缺乏持续动能（RSI中性、ADX低）
        buy_anchor_pos = self._get_anchor_position('BUY')
        if buy_anchor_pos:
            price_reached_edge = price >= buy_anchor_pos['entry_price'] + self.range_flip_multiple * atr
            if price_reached_edge:
                # 双重确认：检查指标是否显示震荡特征（而非趋势突破）
                if self._confirm_ranging_flip('BUY'):
                    logger.warning(f"L2 DGTP: [震荡翻转] BUY锚定仓位达到最大区间利润，双重确认通过，翻转至SELL - 价格: {price:.4f}, "
                                 f"锚定价: {buy_anchor_pos['entry_price']:.4f}, 利润: {price - buy_anchor_pos['entry_price']:.4f}")
                    self._close_all_positions_and_flip('SELL', price, 'RANGE_FLIP_HIGH')
                    return
                else:
                    # 指标显示趋势突破，不执行翻转，继续通过动态对冲保护
                    logger.debug(f"L2 DGTP: [震荡翻转] BUY价格达到边缘，但指标显示趋势突破，不执行翻转 - RSI: {self.macro_context.get_rsi():.2f}, ADX: {self.macro_context.get_adx():.2f}")
        
        sell_anchor_pos = self._get_anchor_position('SELL')
        if sell_anchor_pos:
            price_reached_edge = price <= sell_anchor_pos['entry_price'] - self.range_flip_multiple * atr
            if price_reached_edge:
                # 双重确认：检查指标是否显示震荡特征（而非趋势突破）
                if self._confirm_ranging_flip('SELL'):
                    logger.warning(f"L2 DGTP: [震荡翻转] SELL锚定仓位达到最大区间利润，双重确认通过，翻转至BUY - 价格: {price:.4f}, "
                                 f"锚定价: {sell_anchor_pos['entry_price']:.4f}, 利润: {sell_anchor_pos['entry_price'] - price:.4f}")
                    self._close_all_positions_and_flip('BUY', price, 'RANGE_FLIP_LOW')
                    return
                else:
                    # 指标显示趋势突破，不执行翻转，继续通过动态对冲保护
                    logger.debug(f"L2 DGTP: [震荡翻转] SELL价格达到边缘，但指标显示趋势突破，不执行翻转 - RSI: {self.macro_context.get_rsi():.2f}, ADX: {self.macro_context.get_adx():.2f}")
        
        # ** 2. 宽幅震荡过滤检查（避免在震荡噪音区频繁交易）**
        # 如果价格在通道内（震荡噪音区），且突破未确认，则跳过PYRAMID和HEDGE动作
        is_in_noise_zone = self._is_in_channel(price)
        
        if is_in_noise_zone:
            # 价格在通道内，检查是否有已确认的突破
            breakout_up_confirmed = self._is_channel_breakout_confirmed('UP')
            breakout_down_confirmed = self._is_channel_breakout_confirmed('DOWN')
            
            if not breakout_up_confirmed and not breakout_down_confirmed:
                # 没有已确认的突破，保持静止，不进行任何PYRAMID和HEDGE动作
                logger.debug(f"L2 DGTP: [通道过滤] 价格在震荡噪音区内，保持静止 - 价格: {price:.4f}, "
                           f"突破方向: {self.channel_filter['breakout_direction']}, "
                           f"维持K线: {self.channel_filter['breakout_klines']}")
                return  # 直接返回，不执行后续的PYRAMID和HEDGE逻辑
        
        # ** 3. 波浪动态对冲逻辑（如果没有发生翻转，且突破已确认）**
        # 注意：已移除硬性群体止损机制，风险管理完全通过动态对冲实现
        
        # ==================== BUY 侧逻辑（低多，波峰对冲）====================
        
        if buy_anchor is None and current_sell_vol == 0:
            # 【L2分钟线决策：低多入场意图 - 锚定仓位】
            # L2判断：价格处于低点区域（基于BBands下轨 + step/2），且突破已确认
            # L3执行：接收意图后，使用TICK数据（LRS反转信号）优化进场价格
            # 通道过滤：只有在向下突破已确认时，才允许开多（低点入场）
            if is_low_zone and abs(lrs) < lrs_reverse_threshold:
                # 检查通道过滤：如果启用，需要向下突破已确认
                if self.enable_channel_filter:
                    if not self._is_channel_breakout_confirmed('DOWN'):
                        logger.debug(f"L2 DGTP: [通道过滤] 低多入场被过滤 - 向下突破未确认 "
                                   f"(维持K线: {self.channel_filter['breakout_klines']}/{self.breakout_confirmation_klines})")
                        return
                # [L2意图] 发送低多入场意图，L3 Tick执行层负责最优价进场（VWAP/TWAP优化）
                self._send_dgtp_order('PLACE_ORDER', 'BUY', self.initial_lot, 'R_ANCHOR_BUY_LOW')
                position_id = f'BUY_ANCHOR_{int(time.time() * 1000)}'
                self.dgtp_positions['BUY'].append({
                    'volume': self.initial_lot,
                    'entry_price': price,
                    'type': 'ANCHOR',  # 锚定仓位，不参与短期平仓
                    'id': position_id
                })
                logger.info(f"L2 DGTP: [分钟线决策] 低多入场意图（锚定仓位）→ L3执行 - BUY @ {price:.4f} (低点区域, LRS={lrs:.6f})")
        elif buy_anchor is not None:
            last_pyramid = self._get_last_pyramid_position('BUY')
            pyramid_count = len([pos for pos in self.dgtp_positions['BUY'] if pos.get('type') == 'PYRAMID'])
            
            # A. 波峰操作：平最新加仓 + 补一手空（实现"每个波浪高点出最后加仓那个仓位然后补一手空"）
            # L2判断：最新加仓仓位盈利达到一个网格步长时（波峰）
            # L3执行：接收意图后，使用TICK数据优化出场价格
            if last_pyramid and price > last_pyramid['entry_price'] + step:
                # 1. 平掉最新的加仓仓位（兑现波浪利润）
                close_volume = last_pyramid['volume']
                position_id = last_pyramid.get('id')
                self._send_dgtp_order('CLOSE_POSITION_BY_TYPE', 'BUY', close_volume, 'R_PYRAMID_PROFIT_EXIT')
                # 从本地移除被平仓的PYRAMID
                self.dgtp_positions['BUY'] = [pos for pos in self.dgtp_positions['BUY'] if pos.get('id') != position_id]
                logger.info(f"L2 DGTP: [波峰操作] 平最新加仓仓位 → L3执行 - BUY @ {price:.4f}, "
                           f"入场价: {last_pyramid['entry_price']:.4f}, 盈利: {price - last_pyramid['entry_price']:.4f}")
                
                # 2. 补一手空（反向对冲开仓，固定仓位大小，保持纪律性）
                self._open_hedge_position('SELL', price, 'WAVE_PEAK_HEDGE')
            
            # B. 波谷操作：继续加多（网格补仓）
            # L2判断：价格继续下跌超过步长，且未达最大加仓次数
            # L3执行：接收意图后，使用TICK数据优化补仓价格
            # 🔴 修复：添加边界检查，避免list index out of range
            elif len(self.dgtp_positions['BUY']) > 0 and price < self.dgtp_positions['BUY'][-1]['entry_price'] - step and pyramid_count < self.max_ranging_avg:
                position_id = f'BUY_PYRAMID_{int(time.time() * 1000)}'
                self._send_dgtp_order('PLACE_ORDER', 'BUY', self.initial_lot, 'R_COST_AVERAGE_BUY')
                self.dgtp_positions['BUY'].append({
                    'volume': self.initial_lot,
                    'entry_price': price,
                    'type': 'PYRAMID',  # 加仓仓位，参与波浪捕捉平仓
                    'id': position_id
                })
                logger.info(f"L2 DGTP: [波谷操作] 补仓降低成本意图 → L3执行 - BUY @ {price:.4f}, 锚定价: {buy_anchor:.4f}")
            
            # C. 开启/管理反向对冲（深亏或急跌）
            # L2判断：价格跌破深亏阈值（2.0×ATR）或达到动态对冲步长（动量压缩）
            # L3执行：接收意图后，使用TICK数据优化对冲仓位开仓价格
            else:
                # 计算动态对冲步长（基于动量因子压缩）
                s_hedge, alpha = self._calculate_dynamic_hedge_step('BUY', atr)
                
                # 检查是否达到深亏阈值或动态步长
                if price < buy_anchor - deep_loss_threshold or price < buy_anchor - s_hedge:
                    self._manage_hedge_with_momentum(price, 'SELL', 'BUY', atr)
            
            # D. 对冲管理（在波峰平多后，由这里接管 SELL 对冲仓位）
            # 检查是否有对冲仓位需要管理
            if self._get_total_volume('SELL') > 0:
                self._manage_dynamic_hedge(price, 'SELL', 'BUY', atr)
        
        # ==================== SELL 侧逻辑（高空低平）====================
        
        if sell_anchor is None and current_buy_vol == 0:
            # 【L2分钟线决策：高空入场意图 - 锚定仓位】
            # L2判断：价格处于高点区域（基于BBands上轨 - step/2），且突破已确认
            # L3执行：接收意图后，使用TICK数据（LRS反转信号）优化进场价格
            # 通道过滤：只有在向上突破已确认时，才允许开空（高点入场）
            if is_high_zone and abs(lrs) < lrs_reverse_threshold:
                # 检查通道过滤：如果启用，需要向上突破已确认
                if self.enable_channel_filter:
                    if not self._is_channel_breakout_confirmed('UP'):
                        logger.debug(f"L2 DGTP: [通道过滤] 高空入场被过滤 - 向上突破未确认 "
                                   f"(维持K线: {self.channel_filter['breakout_klines']}/{self.breakout_confirmation_klines})")
                        return
                # [L2意图] 发送高空入场意图，L3 Tick执行层负责最优价进场（VWAP/TWAP优化）
                self._send_dgtp_order('PLACE_ORDER', 'SELL', self.initial_lot, 'R_ANCHOR_SELL_HIGH')
                position_id = f'SELL_ANCHOR_{int(time.time() * 1000)}'
                self.dgtp_positions['SELL'].append({
                    'volume': self.initial_lot,
                    'entry_price': price,
                    'type': 'ANCHOR',  # 锚定仓位，不参与短期平仓
                    'id': position_id
                })
                logger.info(f"L2 DGTP: [分钟线决策] 高空入场意图（锚定仓位）→ L3执行 - SELL @ {price:.4f} (高点区域, LRS={lrs:.6f})")
        elif sell_anchor is not None:
            last_pyramid = self._get_last_pyramid_position('SELL')
            pyramid_count = len([pos for pos in self.dgtp_positions['SELL'] if pos.get('type') == 'PYRAMID'])
            
            # A. 波谷操作：平最新加仓 + 补一手多（实现"每个波浪低点出最后加仓那个仓位然后补一手多"）
            # L2判断：最新加仓仓位盈利达到一个网格步长时（波谷）
            # L3执行：接收意图后，使用TICK数据优化出场价格
            if last_pyramid and price < last_pyramid['entry_price'] - step:
                # 1. 平掉最新的加仓仓位（兑现波浪利润）
                close_volume = last_pyramid['volume']
                position_id = last_pyramid.get('id')
                self._send_dgtp_order('CLOSE_POSITION_BY_TYPE', 'SELL', close_volume, 'R_PYRAMID_PROFIT_EXIT')
                # 从本地移除被平仓的PYRAMID
                self.dgtp_positions['SELL'] = [pos for pos in self.dgtp_positions['SELL'] if pos.get('id') != position_id]
                logger.info(f"L2 DGTP: [波谷操作] 平最新加仓仓位 → L3执行 - SELL @ {price:.4f}, "
                           f"入场价: {last_pyramid['entry_price']:.4f}, 盈利: {last_pyramid['entry_price'] - price:.4f}")
                
                # 2. 补一手多（反向对冲开仓，固定仓位大小，保持纪律性）
                self._open_hedge_position('BUY', price, 'WAVE_TROUGH_HEDGE')
            
            # B. 波峰操作：继续加空（网格补仓）
            # L2判断：价格继续上涨超过步长，且未达最大加仓次数
            # L3执行：接收意图后，使用TICK数据优化补仓价格
            # 🔴 修复：添加边界检查，避免list index out of range
            elif len(self.dgtp_positions['SELL']) > 0 and price > self.dgtp_positions['SELL'][-1]['entry_price'] + step and pyramid_count < self.max_ranging_avg:
                position_id = f'SELL_PYRAMID_{int(time.time() * 1000)}'
                self._send_dgtp_order('PLACE_ORDER', 'SELL', self.initial_lot, 'R_COST_AVERAGE_SELL')
                self.dgtp_positions['SELL'].append({
                    'volume': self.initial_lot,
                    'entry_price': price,
                    'type': 'PYRAMID',  # 加仓仓位，参与波浪捕捉平仓
                    'id': position_id
                })
                logger.info(f"L2 DGTP: [波峰操作] 补仓降低成本意图 → L3执行 - SELL @ {price:.4f}, 锚定价: {sell_anchor:.4f}")
            
            # C. 开启/管理反向对冲（深亏或急涨）
            # L2判断：价格突破深亏阈值（2.0×ATR）或达到动态对冲步长（动量压缩）
            # L3执行：接收意图后，使用TICK数据优化对冲仓位开仓价格
            else:
                # 计算动态对冲步长（基于动量因子压缩）
                s_hedge, alpha = self._calculate_dynamic_hedge_step('SELL', atr)
                
                # 检查是否达到深亏阈值或动态步长
                if price > sell_anchor + deep_loss_threshold or price > sell_anchor + s_hedge:
                    self._manage_hedge_with_momentum(price, 'BUY', 'SELL', atr)
            
            # D. 对冲管理（在波谷平空后，由这里接管 BUY 对冲仓位）
            # 检查是否有对冲仓位需要管理
            if self._get_total_volume('BUY') > 0:
                self._manage_dynamic_hedge(price, 'BUY', 'SELL', atr)
    
    # ==================== 辅助方法：仓位管理 ====================
    
    def _get_last_pyramid_position(self, side: str) -> Optional[Dict[str, Any]]:
        """
        获取最新的 PYRAMID 仓位（最后加仓的那个仓位）
        
        Args:
            side: 方向（'BUY' 或 'SELL'）
            
        Returns:
            最新的 PYRAMID 仓位字典，如果没有则返回 None
        """
        pyramids = [pos for pos in self.dgtp_positions.get(side, []) if pos.get('type') == 'PYRAMID']
        return pyramids[-1] if pyramids else None
    
    def _open_hedge_position(self, side: str, price: float, reason: str):
        """
        打开对冲仓位（保持仓位纪律，固定仓位大小）
        
        注意：仓位大小严格按照初始仓位（initial_lot），不因动量而改变。
        动量只用于压缩触发步长（加速反应），不用于调整仓位大小。
        
        Args:
            side: 方向（'BUY' 或 'SELL'）
            price: 当前价格
            reason: 开仓原因
        """
        hedge_vol = self.initial_lot  # 固定仓位大小，保持纪律性
        position_id = f'{side}_HEDGE_{int(time.time() * 1000)}'
        self._send_dgtp_order('PLACE_ORDER', side, hedge_vol, reason)
        self.dgtp_positions[side].append({
            'volume': hedge_vol,
            'entry_price': price,
            'type': 'HEDGE',
            'id': position_id
        })
        logger.info(f"L2 DGTP: [对冲管理] 开启反向对冲仓位 → L3执行 - {side} @ {price:.4f}, "
                   f"仓位量: {hedge_vol:.4f}, 原因: {reason}")
    
    def _close_all_hedge_positions(self, side: str, volume: float, reason: str):
        """
        平掉某一方向的所有 HEDGE 仓位
        
        Args:
            side: 方向（'BUY' 或 'SELL'）
            volume: 要平仓的总量
            reason: 平仓原因
        """
        if volume > 0:
            self._send_dgtp_order('CLOSE_POSITION_BY_TYPE', side, volume, reason)
            # 只保留非 HEDGE 仓位
            self.dgtp_positions[side] = [pos for pos in self.dgtp_positions[side] if pos.get('type') != 'HEDGE']
            logger.info(f"L2 DGTP: [对冲管理] 平掉 {volume} 的 {side} 对冲仓位 - 原因: {reason}")
    
    # ==================== 动量检测与先行触发 ====================
    
    def _calculate_momentum_factor(self, main_side: str) -> float:
        """
        计算动量因子（α），用于压缩HEDGE触发步长
        
        动量因子范围：0.0（慢速/平稳）到 1.0（高速/急跌急涨）
        
        公式：α = f(momentum, lrs, rsi_change_rate)
        
        Args:
            main_side: 主仓位方向（'BUY' 或 'SELL'）
            
        Returns:
            动量因子（0.0 到 1.0之间）
        """
        momentum = self.micro_context.get_momentum()
        lrs = self.micro_context.get_lrs()
        rsi = self.macro_context.get_rsi()
        
        # 基础动量因子：基于价格变化速度
        momentum_base = min(momentum / self.momentum_threshold, 1.0) if self.momentum_threshold > 0 else 0.0
        
        # LRS因子：基于线性回归斜率的变化速度
        lrs_factor = 0.0
        if main_side == 'BUY':
            # 做多时，LRS快速下降表示急跌
            lrs_factor = min(abs(lrs) / (abs(self.momentum_threshold) * 10), 1.0) if lrs < 0 else 0.0
        else:  # SELL
            # 做空时，LRS快速上升表示急涨
            lrs_factor = min(abs(lrs) / (abs(self.momentum_threshold) * 10), 1.0) if lrs > 0 else 0.0
        
        # RSI因子：基于RSI的变化（从极端值快速回归）
        rsi_factor = 0.0
        if main_side == 'BUY':
            # 做多时，RSI从高位快速下降表示急跌
            rsi_factor = (70 - rsi) / 20.0 if rsi < 70 else 0.0
            rsi_factor = max(0.0, min(1.0, rsi_factor))
        else:  # SELL
            # 做空时，RSI从低位快速上升表示急涨
            rsi_factor = (rsi - 30) / 20.0 if rsi > 30 else 0.0
            rsi_factor = max(0.0, min(1.0, rsi_factor))
        
        # 综合动量因子：取三个因子的最大值（任一指标显示急跌/急涨即可）
        alpha = max(momentum_base, lrs_factor, rsi_factor)
        
        # 限制在配置范围内
        alpha = max(self.momentum_compression_min, min(self.momentum_compression_max, alpha))
        
        return alpha
    
    def _calculate_dynamic_hedge_step(self, main_side: str, atr: float) -> float:
        """
        计算动态对冲步长（基于动量因子压缩基础步长）
        
        公式：S_Hedge = S_Base × (1 - α)
        
        - α = 0.0（慢速）：S_Hedge = S_Base（标准步长）
        - α = 0.5（中速）：S_Hedge = 0.5 × S_Base（步长减半）
        - α = 0.9（高速）：S_Hedge = 0.1 × S_Base（步长压缩到10%）
        
        Args:
            main_side: 主仓位方向（'BUY' 或 'SELL'）
            atr: ATR值
            
        Returns:
            动态对冲步长
        """
        # 基础网格步长
        s_base = self.grid_step_atr * atr
        
        # 计算动量因子
        alpha = self._calculate_momentum_factor(main_side)
        
        # 动态对冲步长 = 基础步长 × (1 - 动量因子)
        s_hedge = s_base * (1.0 - alpha)
        
        return s_hedge, alpha
    
    def _manage_hedge_with_momentum(self, price: float, hedge_side: str, main_side: str, atr: float):
        """
        带动量检测的对冲管理：根据价格变化速度动态压缩触发步长（加速对冲，保持仓位纪律）
        
        核心原则：
        - 动量用于压缩触发步长（加速反应），不用于调整仓位大小
        - 仓位大小严格按照初始仓位（initial_lot），保持纪律性
        
        Args:
            price: 当前价格
            hedge_side: 对冲方向（'BUY' 或 'SELL'）
            main_side: 主仓位方向（'BUY' 或 'SELL'）
            atr: ATR值
        """
        # 检查是否有对冲仓位
        hedge_positions = [pos for pos in self.dgtp_positions.get(hedge_side, []) if pos.get('type') == 'HEDGE']
        
        # 如果没有对冲仓位，计算动态步长并检查是否触发
        if not hedge_positions:
            # 计算动态对冲步长（基于动量因子压缩）
            s_hedge, alpha = self._calculate_dynamic_hedge_step(main_side, atr)
            
            # 检查是否达到动态触发条件
            anchor_price = self._get_anchor_price(main_side)
            if anchor_price is None:
                return
            
            should_trigger = False
            if main_side == 'BUY':
                # 做多时，价格快速下跌
                should_trigger = price < anchor_price - s_hedge
            else:  # SELL
                # 做空时，价格快速上涨
                should_trigger = price > anchor_price + s_hedge
            
            if should_trigger:
                # 根据动量因子记录原因
                if alpha > 0.5:
                    reason = 'MOMENTUM_FAST_HEDGE'
                    logger.warning(f"L2 DGTP: [动量对冲] 快速回调检测 - 动量因子: {alpha:.2f}, "
                                 f"动态步长: {s_hedge:.4f} (压缩率: {(1-alpha)*100:.1f}%)")
                else:
                    reason = 'MOMENTUM_SLOW_HEDGE'
                    logger.info(f"L2 DGTP: [动量对冲] 慢速回调 - 动量因子: {alpha:.2f}, "
                              f"动态步长: {s_hedge:.4f} (压缩率: {(1-alpha)*100:.1f}%)")
                
                # 开启对冲仓位（固定仓位大小，保持纪律性）
                self._open_hedge_position(hedge_side, price, reason)
                return
        
        # 如果已有对冲仓位，继续使用动态对冲管理
        self._manage_dynamic_hedge(price, hedge_side, main_side, atr)
    
    # ==================== 动态对冲管理逻辑 ====================
    
    def _manage_dynamic_hedge(self, price: float, hedge_side: str, main_side: str, atr: float):
        """
        管理反向对冲仓位，根据回调或反转信号动态调整多空比（L2分钟线决策层）
        
        核心逻辑：
        1. 回调判断：价格向主仓位方向回退一个Step → 平空又加一手多（清对冲 + 增主仓）
        2. 反转判断：价格向对冲方向继续运行一个Step → 加一手反向 + 减少倒数第二次加的多（增对冲 + 减主仓）
        
        注意：使用动态步长（基于动量因子压缩），但保持仓位大小固定（纪律性）
        
        Args:
            price: 当前价格
            hedge_side: 对冲方向（'BUY' 或 'SELL'）
            main_side: 主仓位方向（'BUY' 或 'SELL'）
            atr: ATR值
        """
        hedge_positions = [pos for pos in self.dgtp_positions.get(hedge_side, []) if pos.get('type') == 'HEDGE']
        
        # 使用动态步长（基于动量因子压缩）
        grid_step_distance, alpha = self._calculate_dynamic_hedge_step(main_side, atr)
        
        if not hedge_positions:
            return
        
        latest_hedge_pos = hedge_positions[-1]
        hedge_anchor_price = latest_hedge_pos['entry_price']
        
        # 1. 回调判断：价格向主仓位方向回退一个Step → 平空又加一手多（清对冲 + 增主仓）
        # 例如：主 BUY，对冲 SELL。价格从 SELL 锚点上涨一个Step（向主仓位方向回调）
        is_pullback = False
        if hedge_side == 'SELL' and price > hedge_anchor_price + grid_step_distance:
            # 空头对冲，价格上涨（向主仓位BUY方向回调）
            is_pullback = True
        elif hedge_side == 'BUY' and price < hedge_anchor_price - grid_step_distance:
            # 多头对冲，价格下跌（向主仓位SELL方向回调）
            is_pullback = True
        
        if is_pullback:
            # A. 平空（平掉所有对冲仓位）
            hedge_vol = sum(pos['volume'] for pos in hedge_positions)
            self._close_all_hedge_positions(hedge_side, hedge_vol, 'HEDGE_PULLBACK_EXIT')
            
            # B. 又加一手多/空（给主仓位加仓）
            main_pyramid_count = len([pos for pos in self.dgtp_positions[main_side] if pos.get('type') == 'PYRAMID'])
            if main_pyramid_count < self.max_ranging_avg:
                position_id = f'{main_side}_PYRAMID_{int(time.time() * 1000)}'
                self._send_dgtp_order('PLACE_ORDER', main_side, self.initial_lot, f'{main_side}_PULLBACK_ADD')
                self.dgtp_positions[main_side].append({
                    'volume': self.initial_lot,
                    'entry_price': price,
                    'type': 'PYRAMID',
                    'id': position_id
                })
                logger.info(f"L2 DGTP: [回调确认] 平对冲 + 增主仓 → L3执行 - {main_side} @ {price:.4f}")
            return
        
        # 2. 反转判断：价格向对冲仓位方向继续运行一个Step → 加一手反向 + 减少倒数第二次加的多（增对冲 + 减主仓）
        # 例如：主 BUY，对冲 SELL。价格从 SELL 锚点下跌一个Step（向对冲方向继续运行）
        is_reversal = False
        if hedge_side == 'SELL' and price < hedge_anchor_price - grid_step_distance:
            # 价格继续下跌（向对冲方向继续运行）
            is_reversal = True
        elif hedge_side == 'BUY' and price > hedge_anchor_price + grid_step_distance:
            # 价格继续上涨（向对冲方向继续运行）
            is_reversal = True
        
        if is_reversal:
            # 计算动态步长（用于记录日志，但触发已在前面判断）
            s_hedge, alpha = self._calculate_dynamic_hedge_step(main_side, atr)
            
            if alpha > 0.5:
                logger.warning(f"L2 DGTP: [反转确认] 快速反转检测 - 动量因子: {alpha:.2f}, "
                             f"动态步长: {s_hedge:.4f} (压缩率: {(1-alpha)*100:.1f}%)")
            else:
                logger.info(f"L2 DGTP: [反转确认] 慢速反转 - 动量因子: {alpha:.2f}, "
                          f"动态步长: {s_hedge:.4f} (压缩率: {(1-alpha)*100:.1f}%)")
            
            # A. 继续加一手反向（增对冲，固定仓位大小，保持纪律性）
            self._open_hedge_position(hedge_side, price, 'HEDGE_REVERSAL_CONTINUE')
            
            # B. 减少倒数第二次加的多（减主仓 PYRAMID）
            pyramid_positions = [pos for pos in self.dgtp_positions[main_side] if pos.get('type') == 'PYRAMID']
            # 🔴 修复：添加边界检查，避免list index out of range
            if len(pyramid_positions) >= 2:
                # 倒数第二次加仓，即 list[-2]
                try:
                    second_last_pyramid = pyramid_positions[-2]
                    close_volume = second_last_pyramid['volume']
                    position_id = second_last_pyramid.get('id')
                    
                    self._send_dgtp_order('CLOSE_POSITION_BY_TYPE', main_side, close_volume, 'MAIN_POS_REDUCTION_FOR_REVERSAL')
                    
                    # 从本地仓位列表中移除
                    self.dgtp_positions[main_side] = [pos for pos in self.dgtp_positions[main_side] if pos.get('id') != position_id]
                    logger.warning(f"L2 DGTP: [反转确认] 增对冲 + 减主仓 → L3执行 - 平掉倒数第二个 {main_side} PYRAMID仓位 @ {price:.4f}")
                except (IndexError, KeyError) as e:
                    logger.warning(f"L2 DGTP: 访问倒数第二个PYRAMID仓位失败: {e}, pyramid_count={len(pyramid_positions)}")
            return
    
    def _confirm_ranging_flip(self, side: str) -> bool:
        """
        双重确认机制：确认是否应该执行震荡翻转
        
        机制一：价格达到预设区间边缘（已在调用前检查）
        机制二：指标确认市场缺乏持续动能（震荡特征）
        
        Args:
            side: 当前主仓位方向（'BUY' 或 'SELL'）
            
        Returns:
            True: 双重确认通过，可以执行翻转
            False: 指标显示趋势突破，不应执行翻转
        """
        rsi = self.macro_context.get_rsi()
        adx = self.macro_context.get_adx()
        lrs = self.micro_context.get_lrs()
        
        # 获取配置参数
        rsi_neutral_min = float(self.config_manager.get('RANGING', 'RSI_NEUTRAL_MIN', 40.0))
        rsi_neutral_max = float(self.config_manager.get('RANGING', 'RSI_NEUTRAL_MAX', 60.0))
        adx_max_threshold = float(self.config_manager.get('RANGING', 'ADX_MAX_THRESHOLD', 25.0))
        lrs_reverse_threshold = float(self.config_manager.get('RANGING', 'LRS_REVERSE_THRESHOLD', 0.00005))
        
        # 确认条件1：RSI处于中性区域（40-60），显示动量平稳
        rsi_neutral = rsi_neutral_min <= rsi <= rsi_neutral_max
        
        # 确认条件2：ADX低于阈值，显示缺乏强趋势
        adx_low = adx < adx_max_threshold
        
        # 确认条件3：LRS接近0，显示动能衰竭
        lrs_exhausted = abs(lrs) < lrs_reverse_threshold
        
        # 双重确认：至少满足两个条件
        confirmations = sum([rsi_neutral, adx_low, lrs_exhausted])
        
        if confirmations >= 2:
            logger.info(f"L2 DGTP: [双重确认] 通过 - RSI: {rsi:.2f} (中性: {rsi_neutral}), "
                       f"ADX: {adx:.2f} (低趋势: {adx_low}), LRS: {lrs:.6f} (衰竭: {lrs_exhausted})")
            return True
        else:
            logger.debug(f"L2 DGTP: [双重确认] 未通过 - RSI: {rsi:.2f}, ADX: {adx:.2f}, LRS: {lrs:.6f}, "
                        f"确认数: {confirmations}/3")
            return False
    
    def _close_all_positions_and_flip(self, new_anchor_side: str, price: float, reason: str):
        """
        震荡全仓翻转逻辑：平掉所有仓位（BUY, SELL）并开立新的反向 ANCHOR 仓位
        
        这是震荡区间的终极止盈机制，当价格从锚定入场点反向运行达到预设距离时触发。
        实现从低点到高点，再从高点到低点的完整区间套利。
        
        Args:
            new_anchor_side: 新的锚定仓位方向（'BUY' 或 'SELL'）
            price: 当前价格
            reason: 翻转原因
        """
        # 1. 清空所有 BUY 仓位（ANCHOR, PYRAMID, HEDGE）
        buy_volume = self._get_total_volume('BUY')
        if buy_volume > 0:
            command = {
                'action': 'CLOSE_ALL',
                'symbol': self.symbol,
                'type': 'BUY',
                'volume': buy_volume,
                'order_type': 'INTENT',
                'comment': f'{reason}_CLOSE_ALL_BUY',
                'execution_hint': 'TICK_OPTIMIZED'
            }
            self._publish_signal_to_l3_executor(command)
            self.dgtp_positions['BUY'] = []
            logger.info(f"L2 DGTP: [震荡翻转] 清空所有 BUY 仓位 - 总量: {buy_volume:.4f}")
        
        # 2. 清空所有 SELL 仓位（ANCHOR, PYRAMID, HEDGE）
        sell_volume = self._get_total_volume('SELL')
        if sell_volume > 0:
            command = {
                'action': 'CLOSE_ALL',
                'symbol': self.symbol,
                'type': 'SELL',
                'volume': sell_volume,
                'order_type': 'INTENT',
                'comment': f'{reason}_CLOSE_ALL_SELL',
                'execution_hint': 'TICK_OPTIMIZED'
            }
            self._publish_signal_to_l3_executor(command)
            self.dgtp_positions['SELL'] = []
            logger.info(f"L2 DGTP: [震荡翻转] 清空所有 SELL 仓位 - 总量: {sell_volume:.4f}")
        
        # 3. 开立新的反向 ANCHOR 仓位
        position_id = f'{new_anchor_side}_ANCHOR_{int(time.time() * 1000)}'
        self._send_dgtp_order('PLACE_ORDER', new_anchor_side, self.initial_lot, f'{reason}_NEW_ANCHOR')
        self.dgtp_positions[new_anchor_side].append({
            'volume': self.initial_lot,
            'entry_price': price,
            'type': 'ANCHOR',
            'id': position_id
        })
        logger.info(f"L2 DGTP: [震荡翻转] 开立新的反向锚定仓位 → L3执行 - {new_anchor_side} @ {price:.4f}")
    
    def _close_all_positions_by_type(self, side: str, reason: str):
        """
        平掉某一方向的所有仓位或特定类型的仓位（用于清空对冲仓位）
        
        Args:
            side: 方向（'BUY' 或 'SELL'）
            reason: 平仓原因（用于判断是否只清空对冲仓位）
        """
        if not self.dgtp_positions.get(side):
            return
        
        # 如果是清空对冲仓位，只清空 type='HEDGE' 的
        if reason in ['HEDGE_PULLBACK_EXIT', 'HEDGE_CONTINUATION_ADD']:
            hedge_vol = sum(pos['volume'] for pos in self.dgtp_positions[side] if pos.get('type') == 'HEDGE')
            if hedge_vol > 0:
                command = {
                    'action': 'CLOSE_POSITION_BY_TYPE',
                    'symbol': self.symbol,
                    'type': side,
                    'volume': hedge_vol,
                    'order_type': 'INTENT',
                    'comment': reason,
                    'execution_hint': 'TICK_OPTIMIZED'
                }
                self._publish_signal_to_l3_executor(command)
                # 保留非对冲仓位
                self.dgtp_positions[side] = [pos for pos in self.dgtp_positions[side] if pos.get('type') != 'HEDGE']
                logger.info(f"L2 DGTP: [对冲管理] 平掉 {hedge_vol} 的 {side} 对冲仓位 - 原因: {reason}")
        else:
            # 否则全部清仓
            total_vol = self._get_total_volume(side)
            if total_vol > 0:
                command = {
                    'action': 'CLOSE_ALL_POSITIONS_BY_TYPE',
                    'symbol': self.symbol,
                    'type': side,
                    'volume': 0,
                    'order_type': 'INTENT',
                    'comment': reason,
                    'execution_hint': 'TICK_OPTIMIZED'
                }
                self._publish_signal_to_l3_executor(command)
                self.dgtp_positions[side] = []
                logger.info(f"L2 DGTP: [仓位管理] 平掉所有 {side} 仓位 - 原因: {reason}")
    
    # ==================== DGTP策略：趋势模式逻辑 ====================
    
    def _execute_trending_dgtp(self, price: float, trend_side: str):
        """
        趋势模式DGTP逻辑：盈利递增加仓
        
        Args:
            price: 当前价格
            trend_side: 趋势方向（'BUY' 或 'SELL'）
        """
        atr = self.macro_context.get_atr()
        if atr == 0.0:
            return
        
        step = self.grid_step_atr * atr
        
        # 【L2分钟线决策：趋势初始入场意图】
        # L2判断：切换到趋势模式，且当前没有同向仓位
        # L3执行：接收意图后，使用TICK数据优化进场价格
        if not self.dgtp_positions.get(trend_side):
            # [L2意图] 发送趋势初始入场意图，L3 Tick执行层负责最优价进场
            self._send_dgtp_order('PLACE_ORDER', trend_side, self.initial_lot, f'{trend_side}_INITIAL')
            self.dgtp_positions[trend_side].append({
                'volume': self.initial_lot,
                'entry_price': price,
                'type': 'ANCHOR'  # 趋势模式的初始仓位也是锚定仓位
            })
            logger.info(f"L2 DGTP: [分钟线决策] 趋势初始入场意图 → L3执行 - {trend_side} @ {price:.4f}")
            return
        
        # 【L2分钟线决策：盈利递增加仓意图（Pyramiding）】
        # L2判断：价格运行距离 > 当前仓位数量 × 步长（分钟线级别）
        # L3执行：接收意图后，使用TICK数据优化加仓价格（VWAP/TWAP优化）
        worst_entry_price = self._get_anchor_price(trend_side)
        if worst_entry_price is None:
            return
        
        # 计算盈利距离（分钟线级别判断）
        if trend_side == 'BUY':
            profit_distance = price - worst_entry_price
        else:  # SELL
            profit_distance = worst_entry_price - price
        
        # 趋势走得越多，利润越多，越加仓
        pyramid_count = len(self.dgtp_positions[trend_side])
        
        # 仅当价格运行距离超过当前仓位数量乘以步长时才加仓
        if profit_distance > step * pyramid_count and pyramid_count < self.max_pyramid_count:
            # 递增加仓量（例如：第一个加仓量是初始的2倍，第二个是3倍...）
            new_volume = self.initial_lot * (pyramid_count + 1)
            # [L2意图] 发送递增加仓意图，L3 Tick执行层负责最优价进场（VWAP/TWAP优化）
            self._send_dgtp_order('PLACE_ORDER', trend_side, new_volume, f'{trend_side}_PYRAMIDING_{pyramid_count+1}')
            self.dgtp_positions[trend_side].append({
                'volume': new_volume,
                'entry_price': price,
                'type': 'PYRAMID'  # 趋势模式的加仓仓位
            })
            logger.info(f"L2 DGTP: [分钟线决策] 盈利递增加仓意图 → L3执行 - {trend_side} @ {price:.4f}, "
                       f"仓位数: {pyramid_count+1}, 盈利距离: {profit_distance:.4f}")
    
    # ==================== 微观动能刷单叠加模块 ====================
    
    def _execute_scalping_overlay(self, price: float, closed_kline: np.ndarray):
        """
        微观动能刷单叠加模块（Tactical Scalping Overlay V2）
        
        核心设计理念：
        - **微观网格**：负责确定重新入场的位置和间距（即在反弹后哪里再空）
          → 使用斐波那契回撤位（0.382）作为二次入场目标价格
        - **微观动能**：负责确定入场和出场的精确时机（即动能衰竭时平仓）
          → 使用Decay指标判断动能衰竭，使用ΔP判断入场时机
        
        这是将网格的结构性优势和动量的时效性优势结合起来的高级刷单方法。
        
        V2版本核心原则：
        1. **固定仓位**：刷单和对冲仓位固定为0.01手（不随主网格变化）
        2. **目标明确**：只针对主网格中最后建立的、成本最差的多单进行平仓和对冲
        3. **独立结算**：MICRO_HEDGE_SHORT的盈利和亏损必须独立于主网格的PnL进行核算
        
        核心逻辑：
        1. 波浪顶点识别：识别价格上涨到顶点（动能由盛转衰），平掉最后建仓的多单，建MICRO_HEDGE_SHORT空单
        2. 动能衰竭平仓：当空头动能衰竭时（Decay指标 > 0.7），立即平仓MICRO_HEDGE_SHORT（快速止盈）
        3. 等待回撤再空：价格回弹到斐波那契回撤位，再次建仓MICRO_HEDGE_SHORT空单
        
        注意：此模块独立于主网格，不影响主网格的仓位管理和PnL核算
        
        Args:
            price: 当前价格（K线收盘价）
            closed_kline: 刚收盘的K线数据
        """
        if not self.enable_scalping:
            return
        
        atr = self.macro_context.get_atr()
        if atr == 0:
            return
        
        # 获取微观动能指标
        momentum_delta = self.macro_context.get_momentum_delta()  # 动量指标 ΔP
        decay_long = self.macro_context.get_decay_long()  # 多头动能衰竭指标（判断空头动能衰竭）
        decay_short = self.macro_context.get_decay_short()  # 空头动能衰竭指标（判断多头动能衰竭）
        
        # 获取K线数据
        current_high = closed_kline[0]['high']
        current_low = closed_kline[0]['low']
        current_close = closed_kline[0]['close']
        current_open = closed_kline[0]['open']
        
        # 获取主网格仓位（用于判断是否有PYRAMID可平）
        main_buy_pyramids = [pos for pos in self.dgtp_positions.get('BUY', []) if pos.get('type') == 'PYRAMID']
        main_sell_pyramids = [pos for pos in self.dgtp_positions.get('SELL', []) if pos.get('type') == 'PYRAMID']
        
        # 获取刷单仓位（独立管理）
        scalping_buy = self.scalping_positions.get('BUY', [])
        scalping_sell = self.scalping_positions.get('SELL', [])
        
        # ==================== 步骤1：波浪顶点识别与触发（平多建空）- 微观动能：入场时机 ====================
        # V2原则：只针对主网格中最后建立的、成本最差的多单进行平仓和对冲
        # 微观动能负责：使用ΔP判断入场时机（ΔP > 阈值表示急涨到顶点，动能由盛转衰）
        if len(main_buy_pyramids) > 0 and len(scalping_sell) == 0:
            # 获取最后建仓的多单（成本最差的那一笔）
            last_pyramid = main_buy_pyramids[-1]
            
            # 检查波浪顶点条件（微观动能判断）
            # 条件1：动量指标 ΔP > 阈值（价格变化超过1.5倍ATR，表示急涨）
            # 条件2：当前K线是上涨K线（close > open），但出现反转信号
            # 条件3：K线形态出现反转信号（长上影线或吞没形态）
            is_rapid_rise = momentum_delta > self.momentum_entry_threshold
            is_up_candle = current_close > current_open
            # 反转信号：上影线较长（上影线 > 实体 * 0.5）
            has_reversal_signal = ((current_high - max(current_close, current_open)) > 
                                  abs(current_close - current_open) * 0.5)
            
            # 波浪顶点识别：急涨 + 反转信号
            is_wave_peak = is_rapid_rise and is_up_candle and has_reversal_signal
            
            if is_wave_peak:
                # 平掉最后建仓的那一笔多单（成本最差，固定0.01手）
                close_volume = min(last_pyramid['volume'], self.scalping_fixed_lot)  # 只平固定仓位
                position_id = last_pyramid.get('id')
                
                self._send_dgtp_order('CLOSE_POSITION_BY_TYPE', 'BUY', close_volume, 'SCALPING_CLOSE_LAST_PYRAMID')
                logger.info(f"L2 Scalping: [波浪顶点] 平掉最后建仓的多单 → L3执行 - BUY @ {price:.4f}, "
                           f"入场价: {last_pyramid['entry_price']:.4f}, 动量ΔP: {momentum_delta:.4f}, "
                           f"平仓量: {close_volume:.4f} (固定0.01手)")
                
                # 建仓MICRO_HEDGE_SHORT空单（固定0.01手，独立结算）
                scalping_id = f'MICRO_HEDGE_SHORT_{int(time.time() * 1000)}'
                self._send_dgtp_order('PLACE_ORDER', 'SELL', self.scalping_fixed_lot, 'MICRO_HEDGE_SHORT')
                self.scalping_positions['SELL'].append({
                    'volume': self.scalping_fixed_lot,  # 固定0.01手
                    'entry_price': price,
                    'type': 'MICRO_HEDGE_SHORT',  # 标记为刷单仓位
                    'id': scalping_id,
                    'sl_price': price + self.scalping_sl_points,  # 固定止损
                    'entry_time': time.time(),
                    'closed_pyramid_id': position_id,  # 记录被平掉的主网格仓位ID（用于独立结算）
                })
                logger.info(f"L2 Scalping: [波浪顶点] 建仓MICRO_HEDGE_SHORT空单 → L3执行 - SELL @ {price:.4f}, "
                           f"仓位: {self.scalping_fixed_lot:.4f} (固定), 止损: {price + self.scalping_sl_points:.4f}")
        
        # ==================== 步骤2：微观动能衰竭快速平仓MICRO_HEDGE_SHORT - 微观动能：出场时机 ====================
        # 微观动能负责：使用Decay指标判断精确的出场时机（Decay > 0.7表示动能衰竭）
        # 检查MICRO_HEDGE_SHORT空单的平仓条件
        if len(scalping_sell) > 0:
            scalping_sell_pos = scalping_sell[0]  # 刷单通常只有一笔
            
            # 平仓条件1：动能衰竭止盈（微观动能：Decay_Long > 0.7，空头动能衰竭）
            if decay_long > self.decay_exit_threshold:
                close_volume = scalping_sell_pos['volume']
                self._send_dgtp_order('CLOSE_POSITION_BY_TYPE', 'SELL', close_volume, 'SCALPING_DECAY_EXIT')
                
                # 计算独立盈亏（独立于主网格PnL）
                profit = (scalping_sell_pos['entry_price'] - price) * close_volume
                self.scalping_pnl['SELL'] += profit  # 独立结算
                
                logger.info(f"L2 Scalping: [动能衰竭平仓] 平掉MICRO_HEDGE_SHORT空单 → L3执行 - SELL @ {price:.4f}, "
                           f"入场价: {scalping_sell_pos['entry_price']:.4f}, 盈利: {profit:.4f}, "
                           f"Decay: {decay_long:.2f}, 刷单累计PnL: {self.scalping_pnl['SELL']:.4f} (独立结算)")
                
                # 移除刷单仓位
                self.scalping_positions['SELL'] = []
                
                # ==================== 微观网格：计算二次入场位置 ====================
                # 微观网格负责：确定重新入场的位置和间距（斐波那契回撤位）
                fall_range = scalping_sell_pos['entry_price'] - current_low  # 下跌幅度
                self.scalping_state['waiting_reentry'] = 'SELL'
                self.scalping_state['reentry_retracement'] = fall_range
                # 使用斐波那契回撤位（0.382）作为二次入场目标价格（微观网格：结构性优势）
                self.scalping_state['reentry_price'] = current_low + fall_range * self.reentry_retracement_ratio
                logger.info(f"L2 Scalping: [微观网格] 计算二次入场目标价格: {self.scalping_state['reentry_price']:.4f}, "
                           f"回撤比例: {self.reentry_retracement_ratio} (斐波那契), 下跌幅度: {fall_range:.4f}")
            
            # 平仓条件2：固定止损（即时保护）
            elif price >= scalping_sell_pos['sl_price']:
                close_volume = scalping_sell_pos['volume']
                self._send_dgtp_order('CLOSE_POSITION_BY_TYPE', 'SELL', close_volume, 'SCALPING_SL_EXIT')
                
                # 计算独立盈亏（独立于主网格PnL）
                loss = (price - scalping_sell_pos['entry_price']) * close_volume
                self.scalping_pnl['SELL'] += loss  # 独立结算
                
                logger.warning(f"L2 Scalping: [固定止损] 平掉MICRO_HEDGE_SHORT空单 → L3执行 - SELL @ {price:.4f}, "
                             f"入场价: {scalping_sell_pos['entry_price']:.4f}, 亏损: {loss:.4f}, "
                             f"刷单累计PnL: {self.scalping_pnl['SELL']:.4f} (独立结算)")
                
                # 移除刷单仓位
                self.scalping_positions['SELL'] = []
                self.scalping_state['waiting_reentry'] = None
        
        # ==================== 步骤3：等待回撤，重新入场做空（Re-entry）- 微观网格+微观动能 ====================
        # 微观网格：确定入场位置（斐波那契回撤位）
        # 微观动能：确定入场时机（K线反转形态）
        if self.scalping_state['waiting_reentry'] == 'SELL':
            # 检查是否达到二次入场条件
            # 条件1（微观网格）：价格回弹到目标位置（斐波那契回撤位：38.2%或50%）
            is_reentry_price = price >= self.scalping_state['reentry_price']
            
            # 条件2（微观动能）：在该回撤位附近，微观K线再次出现动能衰竭信号（长上影线）
            is_reversal_candle = (current_close < current_open and 
                                 (current_high - max(current_close, current_open)) > 
                                 (max(current_close, current_open) - current_low) * 0.5)  # 上影线较长
            
            if is_reentry_price and is_reversal_candle:
                # 再次建仓MICRO_HEDGE_SHORT空单（固定0.01手，微观网格+微观动能双重确认）
                scalping_id = f'MICRO_HEDGE_SHORT_{int(time.time() * 1000)}'
                self._send_dgtp_order('PLACE_ORDER', 'SELL', self.scalping_fixed_lot, 'MICRO_HEDGE_SHORT_REENTRY')
                self.scalping_positions['SELL'].append({
                    'volume': self.scalping_fixed_lot,  # 固定0.01手
                    'entry_price': price,
                    'type': 'MICRO_HEDGE_SHORT',  # 标记为刷单仓位
                    'id': scalping_id,
                    'sl_price': price + self.scalping_sl_points,
                    'entry_time': time.time()
                })
                logger.info(f"L2 Scalping: [二次入场] 建仓MICRO_HEDGE_SHORT空单 → L3执行 - SELL @ {price:.4f}, "
                       f"仓位: {self.scalping_fixed_lot:.4f} (固定), 止损: {price + self.scalping_sl_points:.4f}, "
                       f"目标价: {self.scalping_state['reentry_price']:.4f} (微观网格), 反转形态: ✓ (微观动能)")
                
                # 重置等待状态
                self.scalping_state['waiting_reentry'] = None
    
    # ==================== DGTP策略：主执行入口 ====================
    
    def _execute_dgtp_strategy(self, price: float):
        """
        DGTP策略主执行入口
        
        根据当前市场模式执行相应的DGTP逻辑
        """
        # 重新加载配置（支持热更新）
        self._load_dgtp_config()
        
        # 根据模式执行相应逻辑
        if self.current_mode == MarketMode.RANGING:
            self._execute_ranging_dgtp(price)
        elif self.current_mode == MarketMode.UPTREND:
            self._execute_trending_dgtp(price, 'BUY')
        elif self.current_mode == MarketMode.DOWNTREND:
            self._execute_trending_dgtp(price, 'SELL')
    
    def _switch_mode(self, new_mode: MarketMode):
        """
        切换市场模式（原子性操作）- DGTP版本
        
        使用锁保护FSM状态切换，确保旧策略清理和新策略启动是同步完成的
        
        Args:
            new_mode: 新的市场模式
        """
        with self._fsm_lock:
            # 如果模式相同或无效，直接返回
            if new_mode == self.current_mode:
                return
            
            logger.info(f"L2 DGTP: 模式切换 - 从 {self.current_mode.name} 切换到 {new_mode.name}")
            
            old_mode = self.current_mode
            
            # ==================== DGTP模式切换：仓位清算逻辑 ====================
            # 1. 从趋势转回震荡：平掉所有加仓仓位，只保留微仓
            if old_mode != MarketMode.RANGING and new_mode == MarketMode.RANGING:
                self._close_all_pyramided_positions()
            
            # 2. 从震荡或反向趋势转为趋势：平掉所有逆势/对冲仓位
            elif new_mode in [MarketMode.UPTREND, MarketMode.DOWNTREND]:
                # UPTREND 平掉所有 SELL 仓位；DOWNTREND 平掉所有 BUY 仓位
                side_to_close = 'SELL' if new_mode == MarketMode.UPTREND else 'BUY'
                self._close_all_positions_by_type(side_to_close, 'TREND_CLEAR_HEDGE')
            
            # 3. 切换状态
            self.current_mode = new_mode
            
            # 4. 重新加载配置（模式切换后）
            self._load_dgtp_config()
            
            # 5. 执行旧策略的平仓逻辑（兼容性）
            if self.current_strategy:
                try:
                    self.current_strategy.on_mode_switch(new_mode)
                except Exception as e:
                    logger.error(f"L2 Core: 策略模式切换回调错误: {e}")
            
            # 6. 创建新策略（兼容性，DGTP策略不依赖这些）
            try:
                if new_mode == MarketMode.RANGING:
                    self.current_strategy = RangingStrategy(self.config_manager, self.symbol)
                elif new_mode == MarketMode.UPTREND:
                    self.current_strategy = UptrendStrategy(self.config_manager, self.symbol)
                elif new_mode == MarketMode.DOWNTREND:
                    self.current_strategy = DowntrendStrategy(self.config_manager, self.symbol)
                else:
                    self.current_strategy = None
                    logger.warning(f"L2 Core: 未知模式: {new_mode}")
                
                # 设置策略的指标上下文
                if self.current_strategy:
                    if hasattr(self.current_strategy, 'set_contexts'):
                        self.current_strategy.set_contexts(self.micro_context, self.macro_context)
                    
                    logger.info(f"L2 DGTP: 策略切换完成 - {old_mode.name} -> {new_mode.name}")
            except Exception as e:
                logger.error(f"L2 Core: 创建新策略失败，回退到旧模式: {e}")
                self.current_mode = old_mode  # 回退
    
    def _send_order_to_l1(self, signal: Signal, price: float):
        """
        L2发送交易指令给L1
        
        Args:
            signal: 交易信号
            price: 当前价格
        """
        # 转换信号为动作
        if signal == Signal.BUY:
            action = 'BUY'
        elif signal == Signal.SELL:
            action = 'SELL'
        elif signal == Signal.CLOSE:
            action = 'CLOSE_ALL'
        else:
            return
        
        # 获取配置参数
        atr_sl_mult = float(self.config_manager.get(self.current_mode.name, 'ATR_SL_MULTIPLIER', 2.0))
        atr = self.macro_context.get_atr()
        
        # 计算SL/TP价格
        if action == 'BUY':
            sl_price = price - atr * atr_sl_mult if atr > 0 else price * 0.99
        elif action == 'SELL':
            sl_price = price + atr * atr_sl_mult if atr > 0 else price * 1.01
        else:
            sl_price = 0.0
        
        # 构建订单指令
        command = {
            'action': action,
            'price': price,
            'volume': 0.01,  # 默认交易量
            'sl': round(sl_price, 5),
            'tp': 0.0  # 简化：不设置止盈
        }
        
        # 推送到Redis List（L1监听此队列）
        try:
            self.r.lpush(L2_ORDER_QUEUE, json.dumps(command))
            logger.info(f"L2 Core: 已发送订单指令到L1 - {action}, price={price}")
        except Exception as e:
            logger.error(f"L2 Core: 发送订单指令失败: {e}")
    
    def _feedback_listener(self):
        """
        后台线程：监听L1发送回来的订单执行反馈
        """
        logger.info("L2 Core: 订单反馈监听线程已启动")
        
        # 使用Redis客户端（文本模式）
        r_text = redis.Redis(
            host=REDIS_CONFIG.get('host', 'localhost'),
            port=REDIS_CONFIG.get('port', 6379),
            db=REDIS_CONFIG.get('db', 0),
            decode_responses=True
        )
        
        while not self.stop_event.is_set():
            try:
                # 阻塞读取L1反馈队列（低延迟BRPOP）
                response = r_text.brpop(L1_FEEDBACK_QUEUE, timeout=0.1)
                
                if response:
                    _, feedback_json = response
                    feedback = json.loads(feedback_json)
                    
                    # 根据反馈更新L2的仓位状态
                    self._update_position_context(feedback)
                    
            except Exception as e:
                logger.error(f"L2 Core: 反馈监听错误: {e}")
                time.sleep(0.05)
        
        logger.info("L2 Core: 订单反馈监听线程已停止")
    
    def _update_position_context(self, feedback: Dict[str, Any]):
        """
        根据订单执行结果更新L2内存中的持仓状态
        
        Args:
            feedback: 订单反馈字典
        """
        status = feedback.get('status')
        action = feedback.get('action')
        
        if status == 'SUCCESS':
            if action in ('BUY', 'SELL'):
                order_id = feedback.get('order_id')
                fill_price = feedback.get('price', feedback.get('fill_price', 0.0))
                logger.info(f"L2 Core: ✅ 订单已成交 - {action}, ID={order_id}, Price={fill_price}")
                
                # 更新持仓状态（简化）
                if self.current_strategy:
                    self.current_strategy.positions[order_id] = {
                        'action': action,
                        'price': fill_price,
                        'time': feedback.get('timestamp', time.time())
                    }
                
                # 更新DGTP仓位（如果订单ID匹配）
                # 注意：这里需要根据订单反馈中的comment来匹配仓位
                comment = feedback.get('comment', '')
                if 'R_INITIAL' in comment or 'R_COST_AVERAGE' in comment or 'R_HEDGE' in comment or '_INITIAL' in comment or '_PYRAMIDING' in comment:
                    # 订单已成交，仓位已在本地管理，这里可以更新order_id
                    pass
            elif action == 'CLOSE_ALL':
                logger.info("L2 Core: ✅ 所有持仓已平仓")
                if self.current_strategy:
                    self.current_strategy.positions.clear()
        else:
            logger.warning(f"L2 Core: ❌ 订单执行失败 - {feedback.get('comment', 'Unknown error')}")
    
    def _push_status_to_l3(self, tick_time_msc: int):
        """
        将当前的策略状态和关键指标推送到L3监控Stream
        
        此操作在L2决策后执行，延迟可接受（O(1) Redis XADD）
        
        Args:
            tick_time_msc: TICK时间戳（毫秒）
        """
        try:
            # 1. 抽取L3需要的数据
            current_price = 0.0
            if self.micro_context.tick_buffer:
                # tick_buffer存储的是(time_msc, price)元组
                current_price = self.micro_context.tick_buffer[-1][1]
            
            # 🔴 修复：使用 get_bbands() 方法获取布林带数据（返回元组）
            bb_upper, bb_mid, bb_lower = self.macro_context.get_bbands()
            
            status_data = {
                'time_msc': tick_time_msc,
                'price': current_price,
                'mode': self.current_mode.name,
                'signal': self.last_signal.name if self.last_signal else 'NONE',
                'lrs': self.micro_context.get_lrs(),
                'density': self.micro_context.get_density(),
                'atr': self.macro_context.get_atr(),
                'bb_upper': bb_upper,
                'bb_lower': bb_lower,
                'bb_mid': bb_mid,
                'adx': self.macro_context.get_adx(),
                'position_count': len(self.current_strategy.positions) if self.current_strategy else 0,
                'dgtp_buy_volume': self._get_total_volume('BUY'),
                'dgtp_sell_volume': self._get_total_volume('SELL'),
            }
            
            # 2. 推送到Redis Stream（JSON格式）
            # 使用文本模式的Redis客户端
            r_text = redis.Redis(
                host=REDIS_CONFIG.get('host', 'localhost'),
                port=REDIS_CONFIG.get('port', 6379),
                db=REDIS_CONFIG.get('db', 0),
                decode_responses=True
            )
            
            r_text.xadd(L3_MONITOR_STREAM, {'status_json': json.dumps(status_data)}, maxlen=1000)
            
        except Exception as e:
            # 仅记录错误，不影响核心流程
            logger.debug(f"L2 Core: 推送L3状态失败: {e}")
    
    def stop(self):
        """停止L2核心决策层"""
        self.stop_event.set()
        self.config_manager.stop()
        
        # 等待线程结束
        if self.data_receiver_thread.is_alive():
            self.data_receiver_thread.join(timeout=2)
        if self.feedback_thread.is_alive():
            self.feedback_thread.join(timeout=2)
        
        logger.info("L2 Core: 已停止")
    
    # ==================== 通用策略状态机决策逻辑 ====================
    
    def _make_decision(self, tick: Dict[str, Any]):
        """
        策略决策核心：基于当前指标和状态机进行状态切换和信号生成
        
        【决策逻辑】
        1. 结合MicroIndicators（瞬时动量/密度）和MacroIndicators（趋势/波动率）
        2. 基于FSM状态切换，生成交易指令
        3. 使用ATR进行动态止损，使用RSI进行超买超卖离场
        
        Args:
            tick: TICK数据字典（包含price, time_msc等）
        """
        try:
            current_price = tick.get('last', tick.get('bid', 0.0))
            if current_price == 0.0:
                return
            
            # 1. 获取核心指标
            # 微观指标
            lrs = self.micro_context.current_lrs
            tick_density = self.micro_context.current_density
            avg_tick_density = self.micro_context.get_avg_density()  # 获取平均密度
            
            # 宏观指标
            rsi = self.macro_context.get_rsi()
            atr = self.macro_context.get_atr()
            bb_upper, bb_mid, bb_lower = self.macro_context.get_bbands()
            
            # 确保ATR有效，用于风控
            if atr is None or atr == 0 or np.isnan(atr):
                atr = 1.0  # 使用默认值或等待初始化
            
            # --- 密度过滤检查（配置化）---
            density_ok = True
            density_reason = ""
            
            if self.DENSITY_FILTER_ACTIVE and self.DENSITY_FILTER_TYPE != 'NONE':
                if avg_tick_density > 0 and tick_density is not None:
                    if self.DENSITY_FILTER_TYPE == 'MOMENTUM_CONFIRM':
                        # 动量突破策略：要求密度高于平均值（确认真实突破）
                        required_density = avg_tick_density * self.DENSITY_AVG_MULTIPLIER
                        if tick_density < required_density:
                            density_ok = False
                            density_reason = f"Density:{tick_density:.2f} < Required:{required_density:.2f} (Avg:{avg_tick_density:.2f} × {self.DENSITY_AVG_MULTIPLIER})"
                    
                    elif self.DENSITY_FILTER_TYPE == 'ARBITRAGE':
                        # 套利策略：要求密度接近平均值（稳定价差）
                        lower_bound = avg_tick_density * (1 - self.DENSITY_ARBITRAGE_RANGE)
                        upper_bound = avg_tick_density * (1 + self.DENSITY_ARBITRAGE_RANGE)
                        if tick_density < lower_bound or tick_density > upper_bound:
                            density_ok = False
                            density_reason = f"Density:{tick_density:.2f} not in range [{lower_bound:.2f}, {upper_bound:.2f}] (Avg:{avg_tick_density:.2f})"
                    
                    elif self.DENSITY_FILTER_TYPE == 'REVERSAL':
                        # 反转策略：低密度时入场（等待价格回归）
                        if tick_density > avg_tick_density * 0.8:  # 密度不能太高
                            density_ok = False
                            density_reason = f"Density:{tick_density:.2f} > Threshold:{avg_tick_density * 0.8:.2f} (低密度策略)"
            
            # --- 状态机逻辑 (FSM) ---
            
            # 状态1: IDLE 或 WAIT_ENTRY
            if self.fsm_state in [StrategyState.IDLE, StrategyState.WAIT_ENTRY]:
                
                # 宏观过滤（例如RSI处于中性区域）
                macro_ok = (rsi is None or (rsi > 30 and rsi < 70))
                
                # 结合宏观过滤、密度过滤和动量信号
                if macro_ok and density_ok:
                    # 瞬时动量突破（微观指标驱动入场）
                    if lrs is not None and lrs > self.ENTRY_LRS_THRESHOLD:
                        # 强劲的短期多头动量
                        reason = f"LRS:{lrs:.4f} > Threshold:{self.ENTRY_LRS_THRESHOLD}"
                        if tick_density is not None:
                            reason += f" & Density:{tick_density:.2f}"
                        self._generate_signal('BUY', current_price, reason, tick)
                        self._transition_to_open('LONG', current_price, atr)
                        
                    elif lrs is not None and lrs < -self.ENTRY_LRS_THRESHOLD:
                        # 强劲的短期空头动量
                        reason = f"LRS:{lrs:.4f} < -Threshold:{self.ENTRY_LRS_THRESHOLD}"
                        if tick_density is not None:
                            reason += f" & Density:{tick_density:.2f}"
                        self._generate_signal('SELL', current_price, reason, tick)
                        self._transition_to_open('SHORT', current_price, atr)
                else:
                    # 记录过滤原因（用于调试）
                    if not macro_ok:
                        logger.debug(f"宏观过滤未通过: RSI={rsi:.2f}")
                    if not density_ok:
                        logger.debug(f"密度过滤未通过: {density_reason}")
                    self.fsm_state = StrategyState.WAIT_ENTRY
                    
            # 状态2: OPEN_LONG
            elif self.fsm_state == StrategyState.OPEN_LONG:
                
                # 止损检查（基于ATR的动态止损）
                stop_loss = self.fsm_position_info['entry_price'] - atr * self.RISK_ATR_MULTIPLIER
                
                # 离场条件A: 价格跌破止损
                if current_price <= stop_loss:
                    self._generate_signal('FLAT', current_price, f"StopLoss hit (SL={stop_loss:.2f}, Entry={self.fsm_position_info['entry_price']:.2f})", tick)
                    self._transition_to_flat()
                    
                # 离场条件B: 宏观指标超买（RSI离场）
                elif rsi is not None and rsi >= self.EXIT_RSI_THRESHOLD:
                    self._generate_signal('FLAT', current_price, f"RSI Overbought ({rsi:.2f} >= {self.EXIT_RSI_THRESHOLD})", tick)
                    self._transition_to_flat()
                
                # 离场条件C: LRS反转（动量衰竭）
                elif lrs is not None and lrs < -self.ENTRY_LRS_THRESHOLD * 0.5:
                    self._generate_signal('FLAT', current_price, f"LRS Reversal ({lrs:.4f} < -{self.ENTRY_LRS_THRESHOLD * 0.5:.4f})", tick)
                    self._transition_to_flat()
            
            # 状态3: OPEN_SHORT
            elif self.fsm_state == StrategyState.OPEN_SHORT:
                
                # 止损检查
                stop_loss = self.fsm_position_info['entry_price'] + atr * self.RISK_ATR_MULTIPLIER
                
                # 离场条件A: 价格突破止损
                if current_price >= stop_loss:
                    self._generate_signal('FLAT', current_price, f"StopLoss hit (SL={stop_loss:.2f}, Entry={self.fsm_position_info['entry_price']:.2f})", tick)
                    self._transition_to_flat()
                    
                # 离场条件B: 宏观指标超卖（RSI离场）
                elif rsi is not None and rsi <= 100 - self.EXIT_RSI_THRESHOLD:  # 例如100-70=30
                    self._generate_signal('FLAT', current_price, f"RSI Oversold ({rsi:.2f} <= {100 - self.EXIT_RSI_THRESHOLD})", tick)
                    self._transition_to_flat()
                
                # 离场条件C: LRS反转（动量衰竭）
                elif lrs is not None and lrs > self.ENTRY_LRS_THRESHOLD * 0.5:
                    self._generate_signal('FLAT', current_price, f"LRS Reversal ({lrs:.4f} > {self.ENTRY_LRS_THRESHOLD * 0.5:.4f})", tick)
                    self._transition_to_flat()
            
            # 状态4: WAIT_CLOSE（如果需要复杂的平仓逻辑，例如等待订单成交）
            # 暂不实现，平仓后直接回到IDLE
            
        except Exception as e:
            logger.error(f"L2 Core: 策略决策错误: {e}")
    
    # --- 状态切换辅助函数 ---
    
    def _transition_to_open(self, side: str, price: float, atr: float):
        """进入持仓状态并记录入场信息"""
        self.fsm_state = StrategyState.OPEN_LONG if side == 'LONG' else StrategyState.OPEN_SHORT
        self.fsm_position_info.update({
            'side': side,
            'entry_price': price,
            'timestamp': time.time(),
            'initial_atr': atr
        })
        logger.warning(f"FSM 状态切换: -> {self.fsm_state} @ {price:.2f} (ATR={atr:.4f})")
    
    def _transition_to_flat(self):
        """进入空仓状态"""
        old_side = self.fsm_position_info.get('side', 'FLAT')
        old_price = self.fsm_position_info.get('entry_price', 0.0)
        self.fsm_state = StrategyState.IDLE
        self.fsm_position_info = {'side': 'FLAT', 'entry_price': 0.0, 'timestamp': 0, 'initial_atr': 0.0}
        logger.warning(f"FSM 状态切换: -> IDLE (平仓 {old_side} @ {old_price:.2f})")
    
    def _generate_signal(self, action: str, price: float, reason: str, tick_data: Optional[Dict[str, Any]] = None):
        """
        生成交易信号，并将其与精炼的决策上下文一起存储
        
        【可视化支持】
        记录完整的决策上下文，包括：
        - 核心指标快照（LRS、RSI、ATR、TICK密度）
        - K线状态（OHLC）
        - FSM状态和仓位信息
        - 时间戳和价格定位
        
        Args:
            action: 交易动作（BUY/SELL/FLAT）
            price: 当前价格
            reason: 信号生成原因
            tick_data: TICK数据字典（可选，用于获取时间戳）
        """
        try:
            # 1. 生成全局唯一的决策ID（用于订单-决策绑定）
            with self.decision_lock:
                self.decision_counter += 1
                decision_id = f"{self.symbol}-{int(time.time() * 1000)}-{self.decision_counter}"
            
            # 2. 获取精炼的决策上下文（Decision Context）
            decision_context = self._get_refined_decision_context(price, tick_data)
            
            # 3. 获取当前M1 K线起始时间
            current_kline_m1 = None
            if hasattr(self.kline_builder, 'kline_states') and 'M1' in self.kline_builder.kline_states:
                current_kline_m1 = self.kline_builder.kline_states['M1'].get('current', {})
            elif hasattr(self.kline_builder, 'current_candle'):
                # 兼容旧版单周期KlineBuilder
                current_kline_m1 = self.kline_builder.current_candle or {}
            
            kline_time_m1 = current_kline_m1.get('time', 0) if current_kline_m1 else 0
            
            # 3. 获取TICK时间戳
            tick_time_ms = 0
            if tick_data:
                tick_time_ms = tick_data.get('time_msc', tick_data.get('time', 0) * 1000)
            elif decision_context.get('tick_time_ms'):
                tick_time_ms = decision_context['tick_time_ms']
            else:
                tick_time_ms = int(time.time() * 1000)
            
            # 4. 构建完整的信号记录（包含决策ID用于订单绑定）
            signal_record = {
                'timestamp': time.time(),
                'tick_time_ms': tick_time_ms,
                'symbol': self.symbol,
                'action': action,  # BUY / SELL / FLAT
                'price': price,
                'reason': reason,
                'kline_time_m1': kline_time_m1,
                'current_state': self.fsm_state,
                'target_state': 'OPEN' if action in ['BUY', 'SELL'] else 'IDLE',
                'fsm_position': self.fsm_position_info.copy(),
                'decision_id': decision_id,  # 唯一决策ID（用于订单绑定）
                'context': decision_context  # 精炼的决策上下文
            }
            
            logger.critical(f"🔔 **TRADE SIGNAL** | {action} @ {price:.2f} | Reason: {reason} | State: {self.fsm_state} | Decision ID: {decision_id}")
            
            # 5. 存储信号记录（推送到Redis Stream供前端可视化）
            self._store_signal_record(signal_record)
            
            # 6. 发送信号到交易执行服务（订单-决策绑定）
            # 注意：如果TradeExecutorService未初始化，则跳过
            if hasattr(self, 'trade_executor_service') and self.trade_executor_service:
                try:
                    exchange_order_id = self.trade_executor_service.send_signal(signal_record)
                    if exchange_order_id:
                        logger.info(f"📤 订单已发送到交易所: {exchange_order_id} | Decision ID: {decision_id}")
                except Exception as e:
                    logger.error(f"发送信号到交易执行服务失败: {e}")
            
            # 7. 可选：将信号推送到订单执行队列（兼容旧接口）
            # try:
            #     L2_ORDER_QUEUE.put(signal_record)
            # except Exception as e:
            #     logger.error(f"推送交易信号到执行队列失败: {e}")
            
        except Exception as e:
            logger.error(f"L2 Core: 生成交易信号失败: {e}")
    
    def _get_refined_decision_context(self, current_price: float, tick_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        获取精炼的决策上下文，只包含前端可视化所需的关键数据
        
        【设计原则】
        - 只记录触发信号的核心指标，避免数据冗余
        - 确保数据格式适合前端JSON渲染
        - 包含足够信息用于决策过程可视化
        
        Args:
            current_price: 当前价格
            tick_data: TICK数据字典（可选）
            
        Returns:
            精炼的决策上下文字典
        """
        try:
            # 宏观指标快照（Macro Snapshot）
            rsi_val = self.macro_context.get_rsi()
            atr_val = self.macro_context.get_atr()
            bb_upper, bb_mid, bb_lower = self.macro_context.get_bbands()
            adx_val = self.macro_context.get_adx()
            
            # 微观指标快照（Micro Snapshot）
            lrs_val = self.micro_context.current_lrs
            density_val = self.micro_context.current_density
            momentum_val = self.micro_context.current_momentum
            
            # 获取当前M1 K线数据
            current_kline_m1 = {}
            if hasattr(self.kline_builder, 'kline_states') and 'M1' in self.kline_builder.kline_states:
                current_kline_m1 = self.kline_builder.kline_states['M1'].get('current', {})
            elif hasattr(self.kline_builder, 'current_candle'):
                current_kline_m1 = self.kline_builder.current_candle or {}
            
            # 获取TICK时间戳
            tick_time_ms = 0
            if tick_data:
                tick_time_ms = tick_data.get('time_msc', tick_data.get('time', 0) * 1000)
            else:
                tick_time_ms = int(time.time() * 1000)
            
            # 组合精炼上下文
            context = {
                # 1. FSM状态信息
                'fsm_state': self.fsm_state,
                'position_side': self.fsm_position_info.get('side', 'FLAT'),
                'position_entry_price': self.fsm_position_info.get('entry_price', 0.0),
                
                # 2. 核心指标数据（精炼，只保留关键值）
                'micro_indicators': {
                    'LRS': round(lrs_val, 5) if lrs_val is not None else None,
                    'TICK_DENSITY': round(density_val, 2) if density_val is not None else None,
                    'MOMENTUM': round(momentum_val, 5) if momentum_val is not None else None,
                },
                'macro_indicators': {
                    'RSI': round(rsi_val, 2) if rsi_val is not None else None,
                    'ATR': round(atr_val, 4) if atr_val is not None else None,
                    'ADX': round(adx_val, 2) if adx_val is not None else None,
                    'BBANDS': {
                        'upper': round(bb_upper, 4) if bb_upper is not None else None,
                        'mid': round(bb_mid, 4) if bb_mid is not None else None,
                        'lower': round(bb_lower, 4) if bb_lower is not None else None,
                    }
                },
                
                # 3. 风控和入场信息
                'risk_management': {
                    'risk_atr_multiplier': self.RISK_ATR_MULTIPLIER,
                    'entry_lrs_threshold': self.ENTRY_LRS_THRESHOLD,
                    'exit_rsi_threshold': self.EXIT_RSI_THRESHOLD,
                },
                
                # 4. K线OHLC（当前M1 K线状态）
                'current_kline_ohlc': {
                    'time': current_kline_m1.get('time', 0),
                    'open': round(current_kline_m1.get('open', 0.0), 4) if current_kline_m1.get('open') else None,
                    'high': round(current_kline_m1.get('high', 0.0), 4) if current_kline_m1.get('high') else None,
                    'low': round(current_kline_m1.get('low', 0.0), 4) if current_kline_m1.get('low') else None,
                    'close': round(current_kline_m1.get('close', 0.0), 4) if current_kline_m1.get('close') else None,
                    'volume': int(current_kline_m1.get('volume', 0)) if current_kline_m1.get('volume') else None,
                },
                
                # 5. 辅助时间信息
                'tick_time_ms': tick_time_ms,
            }
            
            return context
            
        except Exception as e:
            logger.error(f"L2 Core: 获取决策上下文失败: {e}")
            return {
                'fsm_state': self.fsm_state,
                'position_side': 'FLAT',
                'error': str(e)
            }
    
    def _save_closed_kline_to_redis(self, closed_kline: np.ndarray, timeframe: str = 'M1'):
        """
        存储闭合的K线到Redis（供前端查询）
        
        【存储位置】
        - Redis Sorted Set: `kline:{symbol}:{timeframe}` - K线历史数据
        - Redis Pub/Sub: `kline:{symbol}:{timeframe}` - 实时K线更新通知
        
        【数据格式】
        - 符合MT5标准格式：time(秒), open, high, low, close, volume
        - 兼容Lightweight Charts和ECharts要求
        
        Args:
            closed_kline: 闭合的K线NumPy数组（KLINE_DTYPE格式）
            timeframe: 时间周期（'M1', 'M5', 'H1'等）
        """
        try:
            # 转换为字典格式（便于JSON序列化）
            kline_dict = {
                'time': int(closed_kline['time'][0]),  # Unix时间戳（秒）
                'open': float(closed_kline['open'][0]),
                'high': float(closed_kline['high'][0]),
                'low': float(closed_kline['low'][0]),
                'close': float(closed_kline['close'][0]),
                'volume': int(closed_kline['volume'][0]),
                'real_volume': 0  # MT5标准字段，当前未使用
            }
            
            # 使用文本模式的Redis客户端（用于JSON序列化）
            r_text = redis.Redis(
                host=REDIS_CONFIG.get('host', 'localhost'),
                port=REDIS_CONFIG.get('port', 6379),
                db=REDIS_CONFIG.get('db', 0),
                decode_responses=True
            )
            
            # 1. 存储到Redis Sorted Set（历史查询）
            kline_key = f"kline:{self.symbol}:{timeframe.lower()}"  # 转换为小写：M1 -> 1m
            kline_json = json.dumps(kline_dict, ensure_ascii=False)
            
            # 🔴 修复：先删除相同时间戳的旧数据，再添加新数据（避免重复）
            # 使用ZREMRANGEBYSCORE删除相同时间戳的所有数据
            kline_time = kline_dict['time']
            r_text.zremrangebyscore(kline_key, kline_time, kline_time)
            
            # 使用ZADD存储新数据（确保时间戳唯一）
            r_text.zadd(kline_key, {kline_json: kline_time})
            
            # 滚动删除（保留最近2880根，即2天M1数据）
            current_count = r_text.zcard(kline_key)
            if current_count > 2880:
                remove_count = current_count - 2880
                r_text.zremrangebyrank(kline_key, 0, remove_count - 1)
            
            # 2. 发布Pub/Sub通知（供API Server订阅并转发给前端）
            try:
                r_text.publish(
                    f"kline:{self.symbol}:{timeframe.lower()}",
                    kline_json
                )
            except Exception as e:
                logger.debug(f"L2 Core: K线Pub/Sub通知失败（非关键）: {e}")
            
            logger.debug(f"L2 Core: K线已存储到Redis - {timeframe} @ {kline_dict['time']} (O:{kline_dict['open']:.2f} H:{kline_dict['high']:.2f} L:{kline_dict['low']:.2f} C:{kline_dict['close']:.2f})")
            
        except Exception as e:
            logger.error(f"L2 Core: 存储K线到Redis失败: {e}")
    
    def _push_current_kline_to_redis(self, current_kline: Dict[str, Any], timeframe: str = 'M1'):
        """
        🔴 架构重构：此方法已废弃，K线推送由Kline Service负责
        保留此方法以避免破坏现有代码，但不会执行任何操作
        """
        # 🔴 架构重构：K线推送已迁移到Kline Service
        # 此方法保留为空实现，避免破坏现有代码
        return
    
    def _store_signal_record(self, signal_record: Dict[str, Any]):
        """
        存储交易信号记录到Redis Stream（供前端可视化）
        
        【存储位置】
        - Redis Stream: `signal:{symbol}:stream` - 实时信号流
        - Redis Sorted Set: `signal:{symbol}:history` - 历史信号（可选）
        
        Args:
            signal_record: 完整的信号记录字典
        """
        try:
            signal_stream_key = f"signal:{self.symbol}:stream"
            signal_history_key = f"signal:{self.symbol}:history"
            
            # 序列化为JSON
            signal_json = json.dumps(signal_record, ensure_ascii=False)
            
            # 推送到Redis Stream（实时可视化）
            r_text = redis.Redis(
                host=REDIS_CONFIG.get('host', 'localhost'),
                port=REDIS_CONFIG.get('port', 6379),
                db=REDIS_CONFIG.get('db', 0),
                decode_responses=True
            )
            
            # 使用XADD写入Stream，保留最近1000条信号
            r_text.xadd(
                signal_stream_key,
                {'signal_json': signal_json},
                id='*',
                maxlen=1000,
                approximate=True
            )
            
            # 可选：同时保存到Sorted Set（历史查询）
            # 使用时间戳作为score，便于按时间范围查询
            r_text.zadd(
                signal_history_key,
                {signal_json: signal_record['tick_time_ms']}
            )
            
            # 滚动删除（保留最近7天的信号）
            seven_days_ago = signal_record['tick_time_ms'] - (7 * 24 * 60 * 60 * 1000)
            r_text.zremrangebyscore(signal_history_key, '-inf', seven_days_ago)
            
            logger.debug(f"L2 Core: 交易信号已存储到Redis (Stream: {signal_stream_key})")
            
        except Exception as e:
            logger.error(f"L2 Core: 存储交易信号失败: {e}")


# ==================== 测试和演示代码 ====================

if __name__ == '__main__':
    import signal
    import sys
    
    logger.info("=" * 60)
    logger.info("启动L2核心决策层测试")
    logger.info("=" * 60)
    
    # 创建L2核心
    l2_core = L2StrategyCore(symbol="BTCUSDm")
    
    def signal_handler(sig, frame):
        logger.info("\n收到停止信号，正在关闭...")
        l2_core.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # 保持运行
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("\n收到键盘中断，正在关闭...")
        l2_core.stop()

