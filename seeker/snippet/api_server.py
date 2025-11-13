#date: 2025-11-13T17:02:49Z
#url: https://api.github.com/gists/f99a8104b419a7b2f6437a18ce2a8790
#owner: https://api.github.com/users/wangwei334455

"""
Flask后端服务
提供API和WebSocket实时通信
"""
# 🔴 关键：Flask-SocketIO 会自动检测并使用 eventlet（如果可用）
# 但是，如果遇到连接问题，可能需要手动应用 monkey_patch
# 先尝试不手动调用，如果连接失败再启用

# 尝试手动应用 eventlet monkey_patch（如果 eventlet 可用）
try:
    import eventlet
    eventlet.monkey_patch()
    print("✅ eventlet monkey_patch 已应用")
except ImportError:
    print("⚠️ eventlet 未安装，将使用默认异步模式")

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from pathlib import Path
import sys
import json
import redis
import time
from loguru import logger

# 添加路径（必须在导入模块之前）
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(SRC_DIR))

# 现在可以导入trading模块
from src.trading.api.order_engine import OrderEngine

# 定义日志目录
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# 定义标注目录
ANNOTATION_DIR = BASE_DIR / "data" / "annotations"
ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)

# ❌ 已删除：MT5连接器（改用数据采集器 + Redis）
# ❌ 已删除：tick_streamer（改用Redis Stream）

# 创建Flask应用（前端使用独立的Vite开发服务器，不需要Flask提供静态文件）
app = Flask(__name__)

# 设置密钥
app.config['SECRET_KEY'] = "**********"

# 启用CORS
CORS(app)

# 正确初始化SocketIO - 在所有装饰器之前
# 【关键】兼容 socket.io-client 4.x：使用 python-socketio 5.x 的默认配置
# Flask-SocketIO 5.5.1 + python-socketio 5.14.3 支持 Socket.IO 协议 v4
# 【修复】使用 eventlet 异步模式，确保 WebSocket 连接稳定
# 🔴 修复：如果 eventlet 不可用，使用 threading 模式
try:
    import eventlet
    async_mode = 'eventlet'
except ImportError:
    async_mode = 'threading'
    logger.warning("eventlet 未安装，使用 threading 模式（性能较低）")

socketio = SocketIO(
    app, 
    cors_allowed_origins="*",  # 允许所有来源（开发环境）
    logger=False,  # 禁用SocketIO日志
    engineio_logger=False,
    ping_timeout=120,     # 🔴 修复：增加心跳超时时间到120秒，给前端和网络更多宽限，避免ping timeout
    ping_interval=25,      # 🔴 修复：使用标准心跳间隔25秒（与ping_timeout配合，确保有足够时间响应）
    async_mode=async_mode,  # 自动选择异步模式
    allow_upgrades=True,  # 允许协议升级
    transports=['polling', 'websocket'],  # 优先 polling，更稳定
    max_http_buffer_size=1e6,  # 🔴 优化：增加缓冲区大小，避免大数据包丢失
    cors_credentials=True,  # 🔴 优化：允许跨域凭证
)

# 【Redis最佳实践】使用连接池，提高并发性能
redis_pool = redis.ConnectionPool(
    host='localhost',
    port=6379,
    db=0,
    decode_responses=True,
    max_connections=10,  # 连接池大小
    socket_timeout=1.0,  # 🔴 修复：设置socket超时，避免阻塞
    socket_connect_timeout=1.0  # 连接超时
)
redis_client = redis.Redis(connection_pool=redis_pool)

# 全局变量
model = None
model_loaded = False

# 初始化订单引擎
order_engine = OrderEngine()

# 🔴 安全机制：导入环境检查模块（用于订单创建接口）
try:
    from src.trading.utils.env_check import (
        is_production_mode,
        require_production_mode,
        get_env_info,
        log_env_status
    )
    # 启动时记录环境状态
    log_env_status()
except ImportError:
    logger.warning("⚠️ 无法导入环境检查模块，订单创建接口将不进行环境检查")
    def is_production_mode():
        return True  # 降级：允许交易
    def require_production_mode(func_name: str = "创建订单"):
        return True
    def get_env_info():
        return {'env': 'UNKNOWN', 'is_production': True}

# ❌ 旧的tick_streamer模块已删除，现在使用Redis Stream方案


# ==================== 工具函数 ====================

def load_model(model_path: str = None):
    """加载AI模型（暂时禁用）"""
    global model, model_loaded
    logger.warning("AI模型加载已禁用，仅提供数据API")
    model_loaded = False
    return False


def _fill_missing_klines(klines: list, timeframe: str, minutes_per_kline: int) -> list:
    """
    填充缺失的K线（MT5官方最佳实践）
    
    【补空策略】
    1. 检测时间戳连续性
    2. 如果发现空缺，用前一根K线的收盘价填充
    3. 填充的K线：open=close=high=low=前一根收盘价，volume=0
    
    【官方要求】
    - Lightweight Charts要求：时间戳必须连续，不能有缺失
    - 如果市场休市，也应该填充（用前一根收盘价）
    
    Args:
        klines: K线列表（已排序）
        timeframe: 时间周期（'1m', '5m'等）
        minutes_per_kline: 每根K线的分钟数
        
    Returns:
        填充后的K线列表
    """
    if len(klines) < 2:
        return klines
    
    filled_klines = [klines[0]]  # 第一根K线
    
    for i in range(1, len(klines)):
        prev_kline = klines[i - 1]
        current_kline = klines[i]
        
        prev_time = prev_kline.get('time', 0)
        current_time = current_kline.get('time', 0)
        
        # 计算期望的时间间隔（秒）
        expected_interval = minutes_per_kline * 60
        
        # 计算实际时间间隔
        actual_interval = current_time - prev_time
        
        # 如果时间间隔大于期望间隔，说明有缺失
        # 🔴 修复：允许1秒的容差，避免浮点数精度问题
        if actual_interval > expected_interval + 1:  # 允许1秒容差
            # 🔴 修复：更精确的缺失K线数量计算
            # 计算理论上应该有的K线数量，并减去已有的1根（当前K线）
            # 例如：间隔120s，期望60s，应该有2根，缺失1根
            # 例如：间隔180s，期望60s，应该有3根，缺失2根
            kline_count_in_gap = actual_interval // expected_interval
            missing_count = kline_count_in_gap - 1  # 应该有的数量 - 1 (当前K线)
            
            # 填充缺失的K线
            prev_close = prev_kline.get('close', 0)
            if prev_close > 0:  # 确保有有效的收盘价
                for j in range(1, missing_count + 1):
                    missing_time = prev_time + (j * expected_interval)
                    
                    # 创建填充K线（使用前一根收盘价）
                    filled_kline = {
                        'time': missing_time,
                        'open': prev_close,
                        'high': prev_close,
                        'low': prev_close,
                        'close': prev_close,
                        'volume': 0,
                        'real_volume': 0,
                        'is_filled': True  # 标记为填充的K线
                    }
                    filled_klines.append(filled_kline)
                    from datetime import datetime as dt
                    logger.info(f"🔧 填充缺失K线: time={missing_time} ({dt.fromtimestamp(missing_time).strftime('%Y-%m-%d %H:%M:%S')}), price={prev_close:.2f}, 间隔={actual_interval}秒")
            else:
                logger.warning(f"⚠️ 无法填充缺失K线: 前一根收盘价为0, time={prev_time}")
        
        # 添加当前K线
        filled_klines.append(current_kline)
    
    return filled_klines


def format_kline_for_frontend(kline: dict) -> dict:
    """
    将MT5原始格式转换为前端图表格式（ECharts/lightweight-charts）
    
    【数据格式说明】
    - Redis存储：MT5原始格式（time, open, high, low, close, volume, real_volume）
    - 前端展示：转换为图表库需要的格式（可添加timezone等字段）
    - 保留is_filled标记：用于前端识别填充的K线
    
    Args:
        kline: MT5原始格式的K线数据
        
    Returns:
        前端图表格式的K线数据
    """
    formatted = {
        'time': kline.get('time', 0),  # Unix时间戳（秒，UTC）
        'timezone': 'UTC',  # 明确标识时区
        'open': float(kline.get('open', 0)),
        'high': float(kline.get('high', 0)),
        'low': float(kline.get('low', 0)),
        'close': float(kline.get('close', 0)),
        'volume': int(kline.get('volume', 0)),  # tick_volume
        'real_volume': int(kline.get('real_volume', 0)),  # 实际成交量
    }
    
    # 保留is_filled标记（如果存在），用于前端识别填充的K线
    if kline.get('is_filled', False):
        formatted['is_filled'] = True
    
    return formatted


def format_position_for_frontend(pos: dict) -> dict:
    """
    将 MT5 原始格式（下划线命名）的持仓数据转换为前端格式（驼峰命名）。
    
    这是所有对外接口（HTTP API 和 WebSocket）的唯一转换入口。
    遵循"一次转换，多处使用"原则，确保数据格式一致性。
    
    【设计原则】
    1. 单一数据源：所有接口必须调用此函数
    2. 清晰语义：使用 CamelCase 和业务意义的字段名
    3. 时间标准化：统一使用毫秒级 Unix 时间戳
    4. 精度控制：浮点数保留合理精度，避免精度问题
    
    Args:
        pos: MT5 原始格式的持仓数据（包含 ticket, price_open, price_current 等）
        
    Returns:
        dict: 前端期望的统一格式持仓数据（CamelCase 命名）
    """
    import time
    
    # 确保时间戳是毫秒级，前端通常使用毫秒
    current_time_ms = int(time.time() * 1000)
    
    # 获取基础字段
    position_id = str(pos.get('ticket', 0))
    symbol = pos.get('symbol', 'BTCUSDm')
    position_type = pos.get('type', 0)  # 0=BUY/LONG, 1=SELL/SHORT
    
    # 价格字段（MT5 使用下划线命名）
    price_open = float(pos.get('price_open', 0.0))
    price_current = float(pos.get('price_current', 0.0))
    volume = float(pos.get('volume', 0.0))
    
    # 盈亏计算
    profit = float(pos.get('profit', 0.0))
    swap = float(pos.get('swap', 0.0))
    commission = float(pos.get('commission', 0.0))
    unrealized_pnl = profit + swap
    
    # 止损止盈（只有大于0才返回，否则为None）
    sl = float(pos.get('sl', 0.0))
    tp = float(pos.get('tp', 0.0))
    
    # 时间处理：MT5可能提供毫秒时间戳(time_msc/time_update_msc)或秒时间戳(time/time_update)
    time_msc = pos.get('time_msc')
    time_sec = pos.get('time', 0)
    opened_at = time_msc if time_msc else (time_sec * 1000 if time_sec > 0 else 0)
    
    time_update_msc = pos.get('time_update_msc')
    time_update_sec = pos.get('time_update', 0)
    updated_at = (time_update_msc if time_update_msc 
                  else (time_update_sec * 1000 if time_update_sec > 0 else current_time_ms))
    
    # 构建前端格式数据（CamelCase 命名）
    frontend_position = {
        'positionId': position_id,
        'symbol': symbol,
        'side': 'LONG' if position_type == 0 else 'SHORT',
        'volume': volume,
        'entryPrice': price_open,
        'currentPrice': price_current,
        'unrealizedPnL': unrealized_pnl,
        'commission': commission,
        'stopLoss': sl if sl > 0 else None,
        'takeProfit': tp if tp > 0 else None,
        'openedAt': opened_at,
        'updatedAt': updated_at,
    }
    
    # 清理浮点数精度（保留6位小数，足够精度且避免精度问题）
    precision_fields = ['volume', 'entryPrice', 'currentPrice', 'unrealizedPnL', 'commission']
    for key in precision_fields:
        if key in frontend_position and frontend_position[key] is not None:
            frontend_position[key] = round(frontend_position[key], 6)
    
    return frontend_position


def get_kline_data_from_redis(symbol: str = 'BTCUSDm', timeframe: str = '1m', count: int = 100, format_for_frontend: bool = True):
    """
    获取K线数据（从 Redis 读取）
    
    data_puller 通过 gRPC StreamTicks 接收数据并计算 K线，存储到 Redis
    这里直接从 Redis 读取，性能最好（<5ms）
    
    Args:
        symbol: 交易对，如 'BTCUSDm'
        timeframe: 时间周期，支持 '1m', '1h', '1d'
        count: 获取数量，如果为-1或大于等于总数，则获取全部数据
        format_for_frontend: 是否格式化为前端图表格式（默认True）
    """
    try:
        key = f"kline:{symbol}:{timeframe}"
        
        # 🚀 修复：确保获取第一个K线
        # 如果count为-1或非常大，使用0到-1获取全部数据
        if count == -1 or count >= 10000:
            klines = redis_client.zrange(key, 0, -1, withscores=False)
        else:
            # 获取最后count条数据，但确保从第一个开始
            total_count = redis_client.zcard(key)
            if total_count == 0:
                logger.warning(f"Redis中没有K线数据: {key}，请确保 data_puller 正在运行")
                return []
            
            # 如果请求的数量大于等于总数，获取全部
            if count >= total_count:
                # 🔴 修复：限制最大数量，避免一次性读取过多数据阻塞事件循环
                max_count = 10000  # 最多读取10000条
                if total_count > max_count:
                    klines = redis_client.zrange(key, -max_count, -1, withscores=False)
                else:
                    klines = redis_client.zrange(key, 0, -1, withscores=False)
            else:
                # 获取最后count条（包含第一个K线）
                klines = redis_client.zrange(key, -count, -1, withscores=False)
        
        if not klines:
            logger.warning(f"Redis中没有K线数据: {key}，请确保 data_puller 正在运行")
            return []
        
        data = []
        for kline_json in klines:
            kline = json.loads(kline_json)
            if format_for_frontend:
                data.append(format_kline_for_frontend(kline))
            else:
                data.append(kline)
        
        if data:
            from datetime import datetime
            first_time = data[0]['time']
            last_time = data[-1]['time']
            first_dt = datetime.fromtimestamp(first_time)
            last_dt = datetime.fromtimestamp(last_time)
            logger.debug(f"从Redis读取到 {len(data)} 条K线数据，第1条时间: {first_time} ({first_dt.strftime('%Y-%m-%d %H:%M:%S')})，最后一条时间: {last_time} ({last_dt.strftime('%Y-%m-%d %H:%M:%S')})")
        
        return data
        
    except Exception as e:
        logger.error(f"从Redis获取K线数据失败: {e}")
        return []


def get_latest_tick_from_redis(symbol: str = 'BTCUSDm'):
    """
    获取最新TICK数据（从 Redis 读取）
    
    data_puller 通过 gRPC StreamTicks 接收数据并存储到 Redis
    这里直接从 Redis 读取，性能最好（O(1)查询，<0.1ms）
    """
    try:
        key = f"tick:{symbol}:latest"
        tick_json = redis_client.get(key)
        if tick_json:
            return json.loads(tick_json)
        else:
            logger.warning(f"Redis中没有最新TICK数据: {key}，请确保 data_puller 正在运行")
        return None
    except Exception as e:
        logger.error(f"从Redis获取最新TICK失败: {e}")
        return None


def get_tick_history_from_redis(symbol: str = 'BTCUSDm', count: int = 500):
    """
    获取历史TICK数据（从 Redis 读取）
    
    data_puller 通过 gRPC StreamTicks 接收数据并存储到 Redis
    这里直接从 Redis 读取，性能最好（<1ms）
    
    【支持毫秒级精度】保留完整的time_msc，无需去重
    """
    try:
        key = f"tick:{symbol}:realtime"
        ticks = redis_client.zrange(key, -count, -1, withscores=True)
        
        if not ticks:
            logger.warning(f"Redis中没有TICK历史数据: {key}，请确保 data_puller 正在运行")
            return []
        
        result = []
        for tick_json, score in ticks:
            tick = json.loads(tick_json)
            result.append({
                'time': int(score),  # Unix时间戳（秒）
                'time_msc': tick.get('time_msc', int(score * 1000)),  # 毫秒级
                'bid': tick['bid'],
                'ask': tick['ask'],
                'last': tick.get('last', 0.0),
                'volume': tick.get('volume', 0),
            })
        
        logger.debug(f"从Redis读取 {len(result)} 条历史TICK数据（毫秒级）")
        return result
        
    except Exception as e:
        logger.error(f"从Redis获取TICK历史数据失败: {e}")
        return []


# ==================== REST API ====================

@app.route('/')
def index():
    """主页"""
    return send_from_directory(app.template_folder, 'index.html')


@app.route('/api/health', methods=['GET'])
def health_check():
    """
    健康检查端点 - Readiness Probe
    
    【设计原则】
    1. 快速失败：所有依赖检查必须在0.5秒内完成
    2. 依赖解耦：依赖失败不影响服务存活状态
    3. 状态明确：返回详细的依赖健康状态
    
    【状态说明】
    - status='ok': 服务本身运行正常（Liveness）
    - dependencies: 各依赖项的详细状态（Readiness）
    """
    # 🔴 最佳实践：使用ThreadPoolExecutor包装Redis ping，设置严格超时
    redis_status = "Error"
    try:
        import concurrent.futures
        # 使用线程池执行器，设置0.5秒超时（健康检查必须快速）
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(redis_client.ping)
            try:
                redis_connected = future.result(timeout=0.5)  # 0.5秒超时
                redis_status = "OK" if redis_connected else "Error"
            except concurrent.futures.TimeoutError:
                logger.debug("健康检查: Redis ping超时（0.5秒）")
                redis_status = "Timeout"
    except Exception as e:
        logger.debug(f"健康检查: Redis ping失败: {e}")
        redis_status = "Error"
    
    # 🔴 最佳实践：即使依赖失败，服务仍返回200（Readiness Probe）
    # 只有服务本身崩溃时才无法响应（Liveness Probe）
    return jsonify({
        'status': 'ok',  # 服务存活状态（Liveness）
        'service': 'API Server',
        'dependencies': {
            'redis': redis_status,  # 依赖健康状态（Readiness）
            'model_loaded': model_loaded
        }
    }), 200  # 明确返回200状态码


@app.route('/api/klines', methods=['GET'])
def get_klines():
    """
    获取K线数据（MT5官方最佳实践）
    
    【数据获取策略】
    1. 历史K线：直接从MT5获取（使用gRPC GetKlines，MT5官方API）
    2. 当前K线：从Redis读取（用TICK实时生成，因为当前K线还未闭合）
    3. 合并返回：历史K线 + 当前K线
    
    参数:
        symbol: 交易对 (默认: BTCUSDm)
        timeframe: 时间周期，支持 '1m', '1h', '1d' (默认: 1m)
        count: 数量或'all' (默认: all)
    """
    try:
        from datetime import datetime, timedelta
        
        symbol = request.args.get('symbol', 'BTCUSDm')
        timeframe = request.args.get('timeframe', '1m')
        count = request.args.get('count', 'all')  # 支持'all'获取全部数据
        
        # 调试：打印完整请求
        referer = request.headers.get('Referer', 'unknown')
        all_args = dict(request.args)
        logger.info(f"📥 API请求 - 所有参数:{all_args}, 来源:{referer}")
        logger.info(f"📥 解析后 - symbol:{symbol}, timeframe:{timeframe}, count:{count}")
        
        # 解析count参数
        if count == 'all':
            count = 2880  # 默认获取2天M1数据（2880根）
        else:
            count = int(count)
        
        # 定义时间周期映射（供后续使用）
        timeframe_minutes = {
            '1m': 1, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '4h': 240, '1d': 1440
        }
        minutes_per_kline = timeframe_minutes.get(timeframe, 1)
        
        all_klines = []
        
        # 🚀 优化策略：优先从Redis读取（快速响应），然后异步尝试从MT5获取更新
        # 这样可以避免gRPC阻塞导致API超时
        logger.info("优先从Redis读取K线数据（快速响应）")
        redis_klines = get_kline_data_from_redis(symbol, timeframe, count, format_for_frontend=False)
        if redis_klines:
            all_klines = redis_klines
            logger.info(f"✓ 从Redis获取到 {len(all_klines)} 根K线")
        
        # 🚀 策略1: 尝试从MT5获取历史K线（后台异步，不阻塞响应）
        # 注意：如果Redis已有数据，MT5获取失败不影响响应
        try:
            from src.trading.services.grpc_trade_client import get_grpc_client, is_grpc_available
            import concurrent.futures
            
            if is_grpc_available() and len(all_klines) < count * 0.5:  # 如果Redis数据不足，才尝试MT5
                try:
                    client = get_grpc_client()
                    
                    # 计算时间范围
                    import pytz
                    timezone = pytz.timezone("Etc/UTC")
                    to_dt = datetime.now(timezone)
                    from_dt = to_dt - timedelta(minutes=count * minutes_per_kline)
                    
                    to_time = int(to_dt.timestamp())
                    from_time = int(from_dt.timestamp())
                    
                    logger.debug(f"尝试从MT5获取历史K线: {symbol} {timeframe}, 从 {from_time} 到 {to_time}")
                    
                    # 🔴 修复：使用线程池执行器，设置超时保护（2秒），避免阻塞API
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(
                            client.get_klines,
                            symbol=symbol,
                            timeframe=timeframe,
                            from_time=from_time,
                            to_time=to_time,
                            count=count
                        )
                        try:
                            result = future.result(timeout=2.0)
                        except concurrent.futures.TimeoutError:
                            logger.warning(f"从MT5获取历史K线超时（2秒），使用Redis数据")
                            result = None
                    
                    if result and result.get('success') and result.get('klines'):
                        mt5_klines = result['klines']
                        logger.info(f"✓ 从MT5获取到 {len(mt5_klines)} 根历史K线，将合并到现有数据")
                        
                        # 合并MT5数据（去重）
                        mt5_dict = {int(k['time']): {
                            'time': int(k['time']),
                            'open': float(k['open']),
                            'high': float(k['high']),
                            'low': float(k['low']),
                            'close': float(k['close']),
                            'volume': int(k.get('volume', k.get('tick_volume', 0))),
                            'real_volume': int(k.get('real_volume', 0))
                        } for k in mt5_klines}
                        
                        # 合并到all_klines（MT5数据优先）
                        existing_times = {k.get('time', 0) for k in all_klines}
                        for time_key, kline_data in mt5_dict.items():
                            if time_key not in existing_times:
                                all_klines.append(kline_data)
                except Exception as grpc_error:
                    logger.warning(f"从MT5获取K线超时或失败: {grpc_error}，使用Redis数据")
        except Exception as e:
            logger.debug(f"MT5获取K线异常（非关键）: {e}，使用Redis数据")
        
        # 🚀 策略2: 如果Redis也没有数据，返回空数组（快速响应，不阻塞）
        if len(all_klines) == 0:
            logger.warning(f"Redis中没有K线数据: {symbol} {timeframe}，返回空数组")
            return jsonify({
                'success': True,
                'data': [],
                'count': 0,
                'source': 'Redis',
                'filled_count': 0,
                'message': f'暂无K线数据，请确保 data_puller 和 L2 策略核心正在运行'
            })
        
        # 🚀 策略3: 获取当前未闭合的K线（从Redis快照，用TICK实时生成）
        # MT5官方说明：当前K线还未闭合，MT5的copy_rates_range不包含
        # 必须用TICK实时生成当前K线，从L2策略核心推送的current_kline快照获取
        try:
            current_kline_key = f"current_kline:{symbol}:{timeframe}:snapshot"
            current_kline_json = redis_client.get(current_kline_key)
            if current_kline_json:
                current_kline = json.loads(current_kline_json)
                current_time = current_kline.get('time', 0)
                
                # 检查是否已包含在all_klines中（去重）
                if not all_klines or all_klines[-1].get('time') != current_time:
                    all_klines.append(current_kline)
                    logger.debug(f"添加当前未闭合K线: time={current_time}, close={current_kline.get('close', 0):.2f}")
        except Exception as e:
            logger.debug(f"获取当前K线失败（非关键）: {e}")
        
        # 🚀 数据去重和补空（按官方要求）
        if all_klines:
            # 1. 去重：按时间戳去重，保留最新的数据（MT5官方要求：时间戳必须唯一）
            seen_times = {}
            for kline in all_klines:
                kline_time = kline.get('time', 0)
                if kline_time > 0:  # 过滤无效时间戳
                    # 保留最新的数据（后面的数据覆盖前面的，确保时间戳唯一）
                    seen_times[kline_time] = kline
            
            # 转换为列表并按时间排序（Lightweight Charts要求：必须排序）
            unique_klines = list(seen_times.values())
            unique_klines.sort(key=lambda x: x.get('time', 0))
            
            # 2. 补空：检测并填充缺失的K线（MT5官方最佳实践）
            # 🔴 关键：必须在发送给前端之前补空，确保图表连续性
            if len(unique_klines) > 1:
                filled_klines = _fill_missing_klines(
                    unique_klines, 
                    timeframe, 
                    minutes_per_kline
                )
                # 统计填充的K线数量
                filled_count = sum(1 for k in filled_klines if k.get('is_filled', False))
                if filled_count > 0:
                    logger.info(f"✅ 补空完成: 填充了 {filled_count} 根缺失K线")
                unique_klines = filled_klines
            
            # 3. 限制数量
            if count > 0 and len(unique_klines) > count:
                unique_klines = unique_klines[-count:]
            
            all_klines = unique_klines
        
        # 格式化为前端格式
        formatted_klines = []
        for kline in all_klines:
            formatted_klines.append(format_kline_for_frontend(kline))
        
        # 输出发送给前端的数据
        if formatted_klines:
            first = formatted_klines[0]
            last = formatted_klines[-1]
            first_dt = datetime.fromtimestamp(first['time'])
            last_dt = datetime.fromtimestamp(last['time'])
            logger.info(f"📤 API返回 {len(formatted_klines)} 条K线数据:")
            logger.info(f"   第1条: {first['time']} ({first_dt.strftime('%Y-%m-%d %H:%M:%S')}) - 收盘:{first['close']:.2f}")
            logger.info(f"   最后: {last['time']} ({last_dt.strftime('%Y-%m-%d %H:%M:%S')}) - 收盘:{last['close']:.2f}")
        else:
            logger.warning("📤 API返回0条数据")
        
        # 判断数据来源和统计信息
        # 如果从MT5获取到了数据，标记为MT5；否则为Redis
        data_source = 'MT5' if len(all_klines) > 0 and any(
            k.get('time', 0) > 0 and not k.get('is_filled', False) 
            for k in all_klines
        ) else 'Redis'
        
        # 统计填充的K线数量
        filled_count = sum(1 for k in all_klines if k.get('is_filled', False))
        if filled_count > 0:
            logger.info(f"📊 数据统计: 总K线={len(all_klines)}, 填充K线={filled_count}, 来源={data_source}")
        
        return jsonify({
            'success': True,
            'data': formatted_klines,
            'count': len(formatted_klines),
            'source': data_source,
            'filled_count': filled_count  # 填充的K线数量（用于前端显示）
        })
    except Exception as e:
        logger.error(f"❌ /api/klines 路由错误: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'data': [],
            'count': 0,
            'error': str(e)
        }), 500


@app.route('/api/patterns', methods=['GET'])
def get_patterns():
    """获取形态列表"""
    # 临时定义形态类别
    PATTERN_CATEGORIES = {
        0: {"name": "无明显形态", "english": "no_pattern", "type": "neutral"},
        1: {"name": "头肩顶", "english": "head_shoulders_top", "type": "bearish"},
        2: {"name": "头肩底", "english": "head_shoulders_bottom", "type": "bullish"},
    }
    return jsonify({
        'success': True,
        'patterns': PATTERN_CATEGORIES
    })


@app.route('/api/annotations', methods=['GET'])
def get_annotations():
    """获取标注列表"""
    try:
        annotation_files = list(ANNOTATION_DIR.glob("*.json"))
        
        annotations = []
        for ann_file in annotation_files:
            with open(ann_file, 'r', encoding='utf-8') as f:
                ann_data = json.load(f)
                ann_data['id'] = ann_file.stem
                annotations.append(ann_data)
        
        return jsonify({
            'success': True,
            'data': annotations,
            'count': len(annotations)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/annotations', methods=['POST'])
def save_annotation():
    """保存标注"""
    try:
        data = request.json
        
        # 生成文件名
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        pattern_id = data['pattern_id']
        filename = f"annotation_{timestamp}_pattern_{pattern_id}.json"
        
        # 保存文件
        annotation_path = ANNOTATION_DIR / filename
        with open(annotation_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ 标注已保存: {annotation_path}")
        
        return jsonify({
            'success': True,
            'message': '标注保存成功',
            'id': filename.replace('.json', '')
        })
    except Exception as e:
        logger.error(f"保存标注失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/predict', methods=['POST'])
def predict_pattern():
    """形态识别预测"""
    global model, model_loaded
    
    if not model_loaded:
        return jsonify({
            'success': False,
            'error': '模型未加载'
        }), 503
    
    try:
        # 预测功能已移除，此端点保留用于兼容性
        # 实际预测功能应由专门的预测服务提供
        return jsonify({
            'success': False,
            'error': '预测功能暂未实现，请使用其他预测服务'
        }), 501
    except Exception as e:
        logger.error(f"预测失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ==================== WebSocket ====================

# 客户端连接状态管理
clients = {}

@socketio.on('connect')
def handle_connect():
    """客户端连接"""
    # 初始化客户端（last_time=0，接收所有实时tick）
    clients[request.sid] = {
        'last_time': 0,
        'connected_at': time.time()  # 记录连接时间
    }
    logger.info(f"✓ 客户端已连接: {request.sid} | 当前客户端数: {len(clients)}")
    
    emit('connected', {'message': '连接成功'})
    
    # 【最佳实践：像K线一样，从Sorted Set读取历史数据 + Stream读取实时数据】
    # 加载最近5000个TICK点（约1-2小时的数据，取决于市场活跃度）
    try:
        # 【兼容性】优先使用新key，如果为空则尝试旧key
        tick_key_new = 'tick:BTCUSDm'  # ✅ 标准格式
        tick_key_old = 'tick:BTCUSDm:realtime'  # 旧格式（兼容）
        
        # 1. 从Sorted Set读取最近5000条历史TICK
        tick_data = redis_client.zrevrange(tick_key_new, 0, 4999, withscores=True)
        
        # 如果新key没数据，尝试旧key（兼容历史数据）
        if not tick_data:
            tick_data = redis_client.zrevrange(tick_key_old, 0, 4999, withscores=True)
            if tick_data:
                logger.info(f"从旧key读取数据: {tick_key_old}")
        
        if tick_data:
            # 反转顺序（从旧到新）
            tick_data.reverse()
            
            # 解析tick数据
            ticks = []
            for tick_json, score in tick_data:
                try:
                    if isinstance(tick_json, bytes):
                        tick_json = tick_json.decode('utf-8')
                    tick = json.loads(tick_json)
                    ticks.append(tick)
                except Exception as parse_error:
                    logger.warning(f"解析TICK数据失败: {parse_error}")
                    continue
            
            if ticks:
                emit('tick_history', ticks)
                logger.info(f"✓ 从Sorted Set推送 {len(ticks)} 条历史TICK给客户端 {request.sid}")
                # 【关键】不设置last_time，保持为0，确保所有新的实时数据都能推送
        else:
            logger.warning("Sorted Set为空，客户端将只接收实时数据")
    except Exception as e:
        logger.error(f"从Sorted Set读取历史数据失败: {e}")


@socketio.on('disconnect')
def handle_disconnect():
    """客户端断开"""
    if request.sid in clients:
        del clients[request.sid]
        logger.info(f"✗ 客户端已断开并清理: {request.sid} | 剩余客户端: {len(clients)}")


@socketio.on('subscribe_kline')
def handle_subscribe_kline(data):
    """
    订阅K线实时数据
    
    🔴 修复：支持 interval 参数，处理多时间周期订阅
    """
    symbol = data.get('symbol', 'BTCUSDm')
    interval = data.get('interval', '1m')  # 🔴 修复：支持前端传入的interval参数
    
    # 转换前端格式到后端格式：1m -> 1m, 5m -> 5m (保持不变)
    # 后端Redis存储使用小写：kline:{symbol}:1m
    timeframe = interval.lower()
    
    logger.info(f"客户端订阅K线: {symbol} @ {interval}")
    
    # 立即发送当前历史数据（可选，前端已经通过API获取了历史数据）
    # 这里可以选择不发送，或者发送最新的几条作为补充
    # klines = get_kline_data_from_redis(symbol, timeframe, 10, format_for_frontend=True)
    # if klines:
    #     for kline in klines:
    #         kline['symbol'] = symbol
    #         kline['interval'] = interval
    #         emit('kline_update', kline)
    
    # 订阅成功确认（可选）
    emit('kline_subscribed', {'symbol': symbol, 'interval': interval, 'success': True})


@socketio.on('unsubscribe_kline')
def handle_unsubscribe_kline(data):
    """
    取消订阅K线实时数据
    
    🔴 修复：添加 unsubscribe_kline 事件处理
    """
    symbol = data.get('symbol', 'BTCUSDm')
    interval = data.get('interval', '1m')
    logger.info(f"客户端取消订阅K线: {symbol} @ {interval}")
    
    # 取消订阅成功确认（可选）
    emit('kline_unsubscribed', {'symbol': symbol, 'interval': interval, 'success': True})


@socketio.on('subscribe_tick')
def handle_subscribe_tick(data):
    """订阅Tick实时数据"""
    symbol = data.get('symbol', 'BTCUSDm')
    logger.info(f"客户端订阅Tick: {symbol}")


# ==================== 后台任务 ====================

def broadcast_positions_updates():
    """
    定期从MT5获取最新持仓并推送实时更新
    
    功能：
    1. 统一使用 gRPC 从 Windows MT5 中继服务获取最新持仓
    2. 计算浮动盈亏变化
    3. 通过Socket.IO推送 position_update 事件
    """
    from src.trading.services.grpc_trade_client import get_grpc_client, is_grpc_available
    
    last_positions = {}  # 记录上次的持仓，用于检测变化
    last_update_time = 0
    
    logger.info("✓ 持仓实时更新线程已启动（使用 gRPC）")
    
    # 延迟初始化 gRPC 客户端，避免启动时阻塞
    grpc_client = None
    
    while True:
        try:
            current_time = time.time()
            
            # 统一使用 gRPC 获取持仓
            if is_grpc_available():
                try:
                    # 延迟初始化，避免启动时阻塞
                    if grpc_client is None:
                        try:
                            grpc_client = get_grpc_client()
                            # 设置较短的超时，避免阻塞
                            grpc_client.timeout = 2
                        except Exception as e:
                            logger.warning(f"初始化 gRPC 客户端失败: {e}，稍后重试")
                            socketio.sleep(5)
                            continue
                    
                    # 使用超时保护，避免阻塞事件循环
                    try:
                        # 设置超时时间（1秒），快速失败
                        original_timeout = getattr(grpc_client, 'timeout', 10)
                        grpc_client.timeout = 1  # 1秒超时
                        result = grpc_client.get_positions(account_id='', symbol='', ticket=0, magic=0)
                        grpc_client.timeout = original_timeout  # 恢复原始超时
                    except Exception as grpc_error:
                        logger.warning(f"gRPC 获取持仓超时或失败: {grpc_error}")
                        socketio.sleep(5)  # 失败后等待5秒重试
                        continue
                    
                    if result.get('success') and result.get('positions'):
                        positions = result['positions']
                        
                        # 转换为前端格式并推送（使用统一转换函数）
                        for pos in positions:
                            frontend_position = format_position_for_frontend(pos)
                            position_id = frontend_position['positionId']
                            
                            # 检测持仓变化（价格或盈亏变化）
                            last_pos = last_positions.get(position_id)
                            if not last_pos or (
                                last_pos.get('currentPrice') != frontend_position['currentPrice'] or
                                last_pos.get('unrealizedPnL') != frontend_position['unrealizedPnL']
                            ):
                                # 推送持仓更新
                                socketio.emit('position_update', frontend_position)
                                last_positions[position_id] = frontend_position
                                logger.debug(f"推送持仓更新: {position_id}, 盈亏={frontend_position['unrealizedPnL']:.2f}")
                        
                        # 检测已平仓的持仓（从last_positions中删除）
                        current_position_ids = {str(pos.get('ticket', 0)) for pos in positions}
                        for pos_id in list(last_positions.keys()):
                            if pos_id not in current_position_ids:
                                # 持仓已平仓，推送volume=0的更新（前端会删除）
                                socketio.emit('position_update', {
                                    'positionId': pos_id,
                                    'volume': 0  # 前端会删除volume=0的持仓
                                })
                                del last_positions[pos_id]
                                logger.debug(f"持仓已平仓: {pos_id}")
                        
                        socketio.sleep(1)  # gRPC 成功，每1秒更新一次
                        continue
                        
                except Exception as e:
                    logger.warning(f"gRPC 获取持仓失败: {e}")
                    socketio.sleep(5)  # 失败后等待5秒重试
                    continue
            else:
                logger.warning("gRPC 不可用，持仓更新功能暂停")
                socketio.sleep(10)  # gRPC 不可用，等待10秒后重试
                continue
                
        except KeyboardInterrupt:
            raise
        except Exception as e:
            logger.error(f"持仓更新线程异常: {e}")
            import traceback
            logger.error(traceback.format_exc())
            socketio.sleep(5)  # 异常后等待5秒重试


def listen_redis_pubsub():
    """
    监听 Redis Pub/Sub，接收 Windows gRPC 服务推送的实时更新（事件驱动架构）
    
    统一架构：Windows gRPC 服务 → Redis Pub/Sub → 后端 → WebSocket → 前端
    
    订阅频道：
    - tick:{symbol}: TICK数据（高频，来自ZeroMQ或轮询）
    - kline:{symbol}:{timeframe}: K线数据
    - mt5:position_update: 单个持仓更新
    - mt5:positions_update: 持仓列表更新
    - mt5:deal: 成交事件
    - mt5:order_update: 挂单更新
    - mt5:account_info: 账户状态更新
    - mt5:connection_status: 连接状态更新
    - mt5:trade_events: ZeroMQ推送的交易事件
    
    🔴 关键修复：完整的自动重连和指数退避机制，确保线程永不退出
    """
    import json
    
    # 🔴 修复：重连配置常量
    INITIAL_RETRY_DELAY = 1.0  # 初始重试延迟（秒）
    MAX_RETRY_DELAY = 10.0     # 最大重试延迟（秒）
    
    reconnect_delay = INITIAL_RETRY_DELAY
    pubsub = None
    
    # 🔴 修复：辅助函数：创建并订阅PubSub
    def create_and_subscribe_pubsub():
        """创建新的PubSub实例并订阅所有频道"""
        new_pubsub = redis_client.pubsub()
        # 使用模式订阅支持通配符（psubscribe）
        new_pubsub.psubscribe('tick:*')  # 所有TICK数据
        new_pubsub.psubscribe('kline:*')  # 所有K线数据（已闭合）
        new_pubsub.psubscribe('current_kline:*')  # 🚀 当前未闭合K线（实时跳动）
        # 订阅具体频道
        new_pubsub.subscribe(
            'mt5:position_update',
            'mt5:positions_update',
            'mt5:deal',
            'mt5:order_update',
            'mt5:account_info',
            'mt5:connection_status',
            'mt5:trade_events'
        )
        return new_pubsub
    
    # 🔴 修复：无限重试循环，确保线程永不退出
    while True:
        try:
            # 1. 检查Redis连接健康状态
            try:
                redis_client.ping()
            except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError) as e:
                logger.warning(f"Redis Pub/Sub: Redis连接丢失，{reconnect_delay:.1f}秒后重试... 错误: {e}")
                pubsub = None  # 标记pubsub无效
                socketio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, MAX_RETRY_DELAY)  # 指数退避
                continue  # 继续循环，尝试重新连接
            
            # 2. 如果pubsub无效，重新创建
            if pubsub is None:
                try:
                    logger.info("📡 尝试连接 Redis Pub/Sub...")
                    pubsub = create_and_subscribe_pubsub()
                    logger.info("✅ Redis Pub/Sub 连接成功，已订阅所有频道")
                    reconnect_delay = INITIAL_RETRY_DELAY  # 重置延迟
                except Exception as create_error:
                    logger.error(f"❌ Redis Pub/Sub: 创建连接失败: {create_error}，{reconnect_delay:.1f}秒后重试...")
                    pubsub = None
                    socketio.sleep(reconnect_delay)
                    reconnect_delay = min(reconnect_delay * 2, MAX_RETRY_DELAY)  # 指数退避
                    continue
            
            # 3. 🔴 修复：在 eventlet 模式下，使用非阻塞 get_message() 并配合 socketio.sleep()
            # 避免阻塞事件循环，导致无法处理连接关闭事件
            try:
                message = pubsub.get_message(timeout=0.1)  # 100ms超时，避免长时间阻塞
            except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError) as e:
                logger.warning(f"Redis Pub/Sub: 读取消息时连接错误: {e}，重新连接...")
                pubsub = None  # 标记需要重新创建
                socketio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, MAX_RETRY_DELAY)  # 指数退避
                continue
            
            if not message:
                # 没有消息时，让出控制权给事件循环，处理连接关闭等事件
                socketio.sleep(0.01)  # 10ms，让事件循环有机会处理其他事件
                continue
            
            # 处理模式订阅消息（pmessage）和普通消息（message）
            if message['type'] == 'pmessage':
                # 模式订阅消息
                pattern = message['pattern']
                channel = message['channel']
                data_str = message['data']
            elif message['type'] == 'message':
                # 普通订阅消息
                channel = message['channel']
                data_str = message['data']
            else:
                continue
            
            # 解析数据
            # 🔴 修复：current_kline消息可能是字符串格式，需要特殊处理
            if channel.startswith('current_kline:'):
                # 当前K线数据：可能是JSON字符串，需要先解析
                try:
                    if isinstance(data_str, str):
                        data = json.loads(data_str)
                    else:
                        data = data_str
                except (json.JSONDecodeError, TypeError):
                    # 如果解析失败，尝试直接使用
                    data = data_str if isinstance(data_str, dict) else {}
                
                # 🔴 调试：打印接收到的Redis消息（改为INFO级别，便于调试）
                logger.info(f"📥 API Server: 收到Redis current_kline消息: channel={channel}, data={data}")
                handle_current_kline_data(channel, data)
                continue  # 处理完直接继续，避免重复处理
            
            # 其他频道的标准JSON解析
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                logger.warning(f"无法解析JSON数据: {channel}")
                continue
            
            # 🔴 修复：成功处理消息后，重置延迟（连接正常）
            reconnect_delay = INITIAL_RETRY_DELAY
            
            # 根据频道类型分发处理
            if channel.startswith('tick:'):
                # TICK数据：tick:{symbol}
                handle_tick_data(channel, data)
            
            elif channel.startswith('kline:'):
                # K线数据：kline:{symbol}:{timeframe}
                handle_kline_data(channel, data)
            
            elif channel == 'mt5:position_update':
                # 单个持仓更新
                position = data.get('position')
                if position:
                    socketio.emit('position_update', position)
                    logger.debug(f"推送持仓更新: {position.get('positionId')}, 盈亏={position.get('unrealizedPnL', 0):.2f}")
            
            elif channel == 'mt5:positions_update':
                # 持仓结构变化（开/平仓）
                positions = data.get('positions', [])
                logger.debug(f"收到持仓结构更新: {len(positions)}个持仓")
                for pos in positions:
                    socketio.emit('position_update', pos)
            
            elif channel == 'mt5:deal':
                # 新成交
                order_data = data.get('order')
                deal_data = data.get('deal', {})
                if order_data:
                    logger.info(f"收到MT5成交: ticket={deal_data.get('ticket')}, order={deal_data.get('order')}")
                    socketio.emit('order_update', order_data)
                    logger.debug(f"推送订单更新: {order_data.get('orderId')}")
            
            elif channel == 'mt5:order_update':
                # 挂单更新
                order = data.get('order')
                if order:
                    socketio.emit('order_update', order)
                    logger.debug(f"推送挂单更新: {order.get('orderId')}")
            
            elif channel == 'mt5:account_info':
                # 账户状态更新
                account_info = data.get('account_info')
                if account_info:
                    socketio.emit('account_update', account_info)
                    logger.debug(f"推送账户更新: 净值={account_info.get('equity', 0):.2f}")
            
            elif channel == 'mt5:connection_status':
                # 连接状态更新
                status = data.get('status')
                socketio.emit('connection_status', {'status': status})
                logger.info(f"MT5连接状态: {status}")
            
            elif channel == 'mt5:trade_events':
                # ZeroMQ推送的交易事件
                handle_trade_event(data)
            
        except redis.exceptions.ConnectionError as e:
            logger.error(f"❌ Redis Pub/Sub: 连接错误: {e}，{reconnect_delay:.1f}秒后重试...")
            pubsub = None  # 标记需要重新创建
            socketio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, MAX_RETRY_DELAY)  # 指数退避
        except redis.exceptions.TimeoutError as e:
            logger.warning(f"⚠️ Redis Pub/Sub: 超时错误: {e}，{reconnect_delay:.1f}秒后重试...")
            pubsub = None  # 标记需要重新创建
            socketio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, MAX_RETRY_DELAY)  # 指数退避
        except Exception as e:
            # 🔴 修复：捕获所有未知错误，确保线程永不退出
            logger.error(f"❌ Redis Pub/Sub 监听发生未知错误: {e}")
            import traceback
            logger.error(traceback.format_exc())
            pubsub = None  # 标记需要重新创建
            socketio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, MAX_RETRY_DELAY)  # 指数退避


def handle_tick_data(channel: str, data: dict):
    """
    处理TICK数据
    
    Args:
        channel: tick:{symbol}
        data: TICK数据字典
    """
    try:
        symbol = channel.split(':')[-1]
        tick_type = data.get('type', 'tick')
        
        # 更新Redis热缓存（最新TICK）
        tick_key = f'tick:{symbol}'
        tick_json = json.dumps(data)
        redis_client.set(f'{tick_key}:latest', tick_json, ex=60)  # 60秒过期
        
        # 可选：存储到Sorted Set（历史数据）
        if 'time_msc' in data:
            time_msc = data['time_msc']
            redis_client.zadd(tick_key, {tick_json: time_msc})
            # 保留最近10000条
            redis_client.zremrangebyrank(tick_key, 0, -10001)
        
        # 通过WebSocket推送给前端
        # 🔴 修复：统一事件名称为 tick_update（与前端监听一致）
        socketio.emit('tick_update', data)
        
        logger.debug(f"TICK转发: {symbol} @ {data.get('bid', 0):.2f}/{data.get('ask', 0):.2f}")
    
    except Exception as e:
        logger.error(f"处理TICK数据失败: {e}")


def handle_current_kline_data(channel: str, data: dict):
    """
    处理当前未闭合的K线数据（实时跳动，基于TICK更新）
    
    【核心机制】
    - L2策略核心每次收到TICK都会更新当前K线并推送
    - 此函数接收推送，通过WebSocket实时发送给前端
    - 前端接收后更新图表最后一根K线，实现实时跳动
    
    Args:
        channel: current_kline:{symbol}:{timeframe}
        data: 当前K线数据（JSON字符串或字典）
    """
    try:
        # 解析频道：current_kline:{symbol}:{timeframe}
        parts = channel.split(':')
        if len(parts) >= 3:
            symbol = parts[1]
            timeframe = parts[2]
            
            # 🔴 修复：处理数据格式：可能是JSON字符串或字典
            # Redis Pub/Sub返回的数据可能是字符串，需要解析
            if isinstance(data, str):
                try:
                    kline = json.loads(data)
                except json.JSONDecodeError:
                    logger.warning(f"无法解析current_kline JSON: {data[:100]}")
                    return
            elif isinstance(data, dict):
                kline = data
            else:
                logger.warning(f"current_kline数据格式异常: {type(data)}")
                return
            
            if kline and isinstance(kline, dict) and kline.get('time', 0) > 0:
                # 🚀 转换时间周期格式：后端使用 m1/M1，前端使用 1m
                # 例如：m1 -> 1m, m5 -> 5m, h1 -> 1h, d1 -> 1d
                def convert_timeframe_to_interval(tf: str) -> str:
                    """将后端时间周期格式转换为前端格式"""
                    if not tf:
                        return '1m'
                    tf_lower = tf.lower()
                    # 处理 M1, M5, H1, D1 等格式
                    if tf_lower.startswith('m'):
                        # m1 -> 1m, m5 -> 5m
                        minutes = tf_lower[1:] if len(tf_lower) > 1 else '1'
                        return f"{minutes}m"
                    elif tf_lower.startswith('h'):
                        # h1 -> 1h, h4 -> 4h
                        hours = tf_lower[1:] if len(tf_lower) > 1 else '1'
                        return f"{hours}h"
                    elif tf_lower.startswith('d'):
                        # d1 -> 1d
                        days = tf_lower[1:] if len(tf_lower) > 1 else '1'
                        return f"{days}d"
                    else:
                        # 默认返回 1m
                        return '1m'
                
                frontend_interval = convert_timeframe_to_interval(timeframe)
                
                # 🚀 确保数据格式符合Lightweight Charts标准
                kline_dict = {
                    'symbol': symbol,
                    'interval': frontend_interval,  # 🚀 修复：使用前端格式（1m而不是m1）
                    'timeframe': timeframe,  # 保留原始格式供调试
                    'time': int(kline.get('time', 0)),  # Unix时间戳（秒）
                    'open': float(kline.get('open', 0)),
                    'high': float(kline.get('high', 0)),
                    'low': float(kline.get('low', 0)),
                    'close': float(kline.get('close', 0)),  # 实时TICK价格
                    'volume': int(kline.get('volume', 0)),
                    'openTime': int(kline.get('time', 0)) * 1000,  # 兼容IBackendKLine格式（毫秒）
                    'closeTime': int(kline.get('time', 0)) * 1000 + 60,  # 假设1分钟K线
                    'is_closed': kline.get('is_closed', False)  # 标记未闭合
                }
                
                # 通过WebSocket推送给前端（实时跳动）
                # 🔴 修复：根据Flask-SocketIO文档，在后台任务中直接使用socketio.emit()会发送给所有客户端
                # 参考：https://flask-socketio.readthedocs.io/en/latest/getting_started.html#emitting-from-background-tasks
                if len(clients) > 0:
                    socketio.emit('kline_update', kline_dict)  # 🔴 修复：移除broadcast参数，Flask-SocketIO不支持
                    logger.debug(f"📤 推送当前K线跳动: symbol={symbol}, timeframe={timeframe}, interval={frontend_interval}, "
                               f"time={kline_dict['time']}, close={kline_dict['close']:.2f}, 客户端数={len(clients)}")
                else:
                    logger.debug(f"⚠️ 无客户端连接，跳过K线推送: symbol={symbol}, interval={frontend_interval}")
    
    except Exception as e:
        logger.error(f"处理当前K线数据失败: {e}")


def handle_kline_data(channel: str, data: dict):
    """
    处理K线数据（从Redis Pub/Sub接收）
    
    Args:
        channel: kline:{symbol}:{timeframe}
        data: K线数据（可能是JSON字符串或字典）
    """
    try:
        # 解析频道：kline:{symbol}:{timeframe}
        parts = channel.split(':')
        if len(parts) >= 3:
            symbol = parts[1]
            timeframe = parts[2]
            
            # 处理数据格式：可能是JSON字符串或字典
            if isinstance(data, str):
                try:
                    kline = json.loads(data)
                except:
                    kline = {'kline': json.loads(data)}
            else:
                kline = data.get('kline') if 'kline' in data else data
            
            if kline and isinstance(kline, dict):
                # 🚀 确保数据格式符合MT5和Lightweight Charts标准
                kline_dict = {
                    'time': int(kline.get('time', 0)),  # Unix时间戳（秒）
                    'open': float(kline.get('open', 0)),
                    'high': float(kline.get('high', 0)),
                    'low': float(kline.get('low', 0)),
                    'close': float(kline.get('close', 0)),
                    'volume': int(kline.get('volume', 0)),
                    'real_volume': int(kline.get('real_volume', 0))
                }
                
                # 存储到Redis Sorted Set（如果尚未存储）
                kline_key = f'kline:{symbol}:{timeframe}'
                kline_json = json.dumps(kline_dict, ensure_ascii=False)
                
                # 🔴 修复：先删除相同时间戳的旧数据，再添加新数据（避免重复）
                kline_time = kline_dict['time']
                redis_client.zremrangebyscore(kline_key, kline_time, kline_time)
                
                # 使用ZADD存储新数据（确保时间戳唯一）
                redis_client.zadd(kline_key, {kline_json: kline_time})
                
                # 保留最近2880根（2天M1数据）
                current_count = redis_client.zcard(kline_key)
                if current_count > 2880:
                    remove_count = current_count - 2880
                    redis_client.zremrangebyrank(kline_key, 0, remove_count - 1)
                
                # 通过WebSocket推送给前端（Lightweight Charts格式）
                # 🔴 修复：根据Flask-SocketIO文档，在后台任务中直接使用socketio.emit()会发送给所有客户端
                if len(clients) > 0:
                    socketio.emit('kline_update', {
                        'symbol': symbol,
                        'interval': timeframe,  # 使用interval保持一致性
                        'timeframe': timeframe,
                        'time': kline_dict['time'],
                        'open': kline_dict['open'],
                        'high': kline_dict['high'],
                        'low': kline_dict['low'],
                        'close': kline_dict['close'],
                        'volume': kline_dict['volume'],
                        'openTime': kline_dict['time'] * 1000,  # 兼容IBackendKLine格式（毫秒）
                        'closeTime': kline_dict['time'] * 1000 + 60  # 假设1分钟K线
                    })  # 🔴 修复：移除broadcast参数，Flask-SocketIO不支持
                    logger.debug(f"K线转发: {symbol} {timeframe} @ {kline_dict['time']} (O:{kline_dict['open']:.2f} C:{kline_dict['close']:.2f}), 客户端数={len(clients)}")
                else:
                    logger.debug(f"⚠️ 无客户端连接，跳过K线转发: {symbol} {timeframe}")
    
    except Exception as e:
        logger.error(f"处理K线数据失败: {e}", exc_info=True)


def handle_trade_event(data: dict):
    """
    处理ZeroMQ推送的交易事件
    
    Args:
        data: 交易事件数据
    """
    try:
        trade_type = data.get('trade_type')
        order_ticket = data.get('order_ticket')
        position_ticket = data.get('position_ticket')
        
        logger.info(f"收到ZeroMQ交易事件: type={trade_type}, order={order_ticket}, position={position_ticket}")
        
        # 根据交易类型处理
        if trade_type in [2, 3, 4]:  # TRADE_TRANSACTION_DEAL_ADD, POSITION, ORDER_ADD
            # 触发持仓/订单查询更新
            socketio.emit('trade_event', data)
            
            # 可选：通知前端刷新订单/持仓列表
            socketio.emit('refresh_orders')
            socketio.emit('refresh_positions')
    
    except Exception as e:
        logger.error(f"处理交易事件失败: {e}")


def listen_order_feedback():
    """
    监听订单执行反馈队列，推送订单和持仓更新
    
    功能：
    1. 监听Redis反馈队列 l1:order:feedback
    2. 当订单执行成功后，推送 order_update 和 position_update 事件
    """
    logger.info("✓ 订单反馈监听线程已启动")
    
    while True:
        try:
            # 从Redis List读取反馈（非阻塞）
            feedback_json = redis_client.lpop('l1:order:feedback')
            
            if feedback_json:
                feedback = json.loads(feedback_json)
                logger.info(f"收到订单反馈: {feedback}")
                
                action = feedback.get('action')
                status = feedback.get('status')
                order_id = feedback.get('order_id')
                
                if status == 'SUCCESS':
                    # 订单执行成功，推送订单更新
                    if action in ['BUY', 'SELL']:
                        # 开仓成功，需要获取订单详情
                        try:
                            # 从OrderEngine获取订单详情
                            all_orders = order_engine.get_all_orders()
                            order = next((o for o in all_orders if str(o.get('ticket')) == str(order_id)), None)
                            
                            if order:
                                # 转换为前端格式
                                frontend_order = {
                                    'orderId': str(order.get('ticket', order_id)),
                                    'symbol': order.get('symbol', 'BTCUSDm'),
                                    'side': 'BUY' if order.get('type', 0) == 0 else 'SELL',
                                    'type': 'MARKET',
                                    'volume': order.get('volume_initial', order.get('volume', 0.0)),
                                    'price': order.get('price_open', 0.0),
                                    'status': 'FILLED',  # 市价单立即成交
                                    'createdAt': order.get('time_setup', 0) * 1000,
                                    'updatedAt': int(time.time()) * 1000
                                }
                                
                                # 推送订单更新
                                socketio.emit('order_update', frontend_order)
                                logger.info(f"推送订单更新: {frontend_order['orderId']}")
                                
                                # 如果是开仓，同时推送持仓更新
                                if order.get('state') == 4:  # ORDER_STATE_FILLED
                                    positions = order_engine.get_all_positions()
                                    position = next((p for p in positions if str(p.get('ticket')) == str(order_id)), None)
                                    
                                    if position:
                                        frontend_position = format_position_for_frontend(position)
                                        socketio.emit('position_update', frontend_position)
                                        logger.info(f"推送持仓更新: {frontend_position['positionId']}")
                        
                        except Exception as e:
                            logger.error(f"处理订单反馈失败: {e}")
                    
                    elif action == 'CLOSE':
                        # 平仓成功，推送持仓更新（volume=0）
                        socketio.emit('position_update', {
                            'positionId': str(order_id),
                            'volume': 0  # 前端会删除
                        })
                        logger.info(f"推送平仓更新: {order_id}")
            
            socketio.sleep(0.1)  # 100ms检查一次
            
        except Exception as e:
            logger.error(f"订单反馈监听异常: {e}")
            socketio.sleep(1)


def broadcast_realtime_data():
    """
    【Flask-SocketIO + Redis Stream 最佳实践】
    
    参考：
    - Flask-SocketIO: https://flask-socketio.readthedocs.io/en/latest/getting_started.html#background-tasks
    - Redis Stream: https://redis.io/docs/data-types/streams-tutorial/
    
    关键要点：
    1. ✅ 使用 socketio.start_background_task() 启动（不用 threading.Thread）
    2. ✅ 使用 Consumer Group 确保消息不丢失
    3. ✅ 使用非阻塞或短时间阻塞（避免阻塞事件循环）
    4. ✅ 使用 socketio.sleep() 而不是 time.sleep()
    """
    
    # 🔴 架构修复：API Server应该消费已验证流（与L2策略核心同级）
    # 原始流: tick:BTCUSDm:stream (Data Puller写入)
    # 已验证流: tick:BTCUSDm:validated:stream (Data Integrity Service写入)
    # L2策略核心和API Server都应该消费已验证流，确保数据完整性
    stream_key = 'tick:BTCUSDm:validated:stream'  # ✅ 使用已验证流
    stream_key_old = 'tick:BTCUSDm:stream'  # 兼容旧格式（原始流）
    group_name = 'backend_broadcast'
    consumer_name = 'worker_1'
    
    # 创建消费者组（如果不存在）
    try:
        redis_client.xgroup_create(stream_key, group_name, id='$', mkstream=True)
        logger.info(f"✓ 创建消费者组: {group_name}")
    except Exception as e:
        if 'BUSYGROUP' not in str(e):
            logger.error(f"创建消费者组失败: {e}")
    
    logger.info(f"✓ Stream推送已启动: {stream_key} ({group_name}/{consumer_name})")
    logger.info(f"  - 消费已验证流（与L2策略核心同级）")
    logger.info(f"  - 确保数据完整性（seq、checksum已验证）")
    
    last_kline_push = 0  # 记录上次推送K线的时间
    current_stream_key = stream_key  # 当前使用的Stream key
    fallback_attempted = False  # 是否已尝试fallback
    no_data_count = 0  # 🔴 修复：记录连续无数据次数，避免频繁切换
    
    while True:
        try:
            # 【最佳实践】使用短时间阻塞（10ms），避免长时间阻塞事件循环
            # 🔴 修复：在 eventlet 模式下，缩短阻塞时间并配合 socketio.sleep()，避免阻塞事件循环导致无法接受新连接
            try:
                streams = redis_client.xreadgroup(
                    group_name, 
                    consumer_name,
                    {current_stream_key: '>'},
                    count=10,
                    block=10  # 🔴 修复：缩短到 10ms，避免阻塞事件循环
                )
            except redis.exceptions.ConnectionError as e:
                logger.warning(f"Redis Stream读取连接错误: {e}")
                socketio.sleep(1)
                continue
            except Exception as e:
                logger.error(f"Redis Stream读取异常: {e}")
                socketio.sleep(0.1)
                continue
            
            # 🔴 修复：如果有数据，重置计数器
            if streams:
                no_data_count = 0
            else:
                no_data_count += 1
                # 🔴 修复：没有数据时，让出控制权给事件循环，处理连接关闭等事件
                socketio.sleep(0.01)  # 10ms，让事件循环有机会处理其他事件
            
            # 【兼容性】如果新Stream长时间没数据（连续100次无数据，约5秒），尝试切换到旧Stream
            # 🔴 修复：避免频繁切换Stream导致日志刷屏和性能问题
            if not streams and not fallback_attempted and no_data_count >= 100:
                # 检查旧Stream是否有数据
                try:
                    old_stream_info = redis_client.xinfo_stream(stream_key_old)
                    if old_stream_info.get('length', 0) > 0:
                        logger.info(f"切换到旧Stream: {stream_key_old} (连续{no_data_count}次无数据)")
                        redis_client.xgroup_create(stream_key_old, group_name, id='$', mkstream=True)
                        current_stream_key = stream_key_old
                        fallback_attempted = True
                        no_data_count = 0  # 重置计数器
                except Exception as e:
                    # 🔴 修复：只记录一次错误，避免日志刷屏
                    if not hasattr(broadcast_realtime_data, '_fallback_error_logged'):
                        logger.warning(f"检查旧Stream失败: {e}")
                        broadcast_realtime_data._fallback_error_logged = True
                    fallback_attempted = True  # 标记已尝试，避免重复检查
            
            if streams:
                for stream_name, messages in streams:
                    for message_id, data in messages:
                        try:
                            # 解析TICK数据（Redis返回的key可能是str或bytes）
                            # Redis Stream 可能使用 'data' 或 'value' 字段
                            tick_json = data.get(b'data') or data.get('data') or data.get(b'value') or data.get('value')
                            if not tick_json:
                                logger.error(f"Stream数据缺少data/value字段: {data}")
                                # 🔴 修复：即使数据格式错误也要ACK，避免pending堆积
                                redis_client.xack(current_stream_key, group_name, message_id)
                                continue
                            
                            # 如果是bytes，需要decode
                            if isinstance(tick_json, bytes):
                                tick_json = tick_json.decode('utf-8')
                            
                            tick = json.loads(tick_json)
                            
                            # 验证必需字段（MT5最佳实践）
                            if not all(k in tick for k in ['time_msc', 'bid', 'ask']):
                                logger.warning(f"TICK数据格式不完整: {tick}")
                                # 🔴 修复：即使数据不完整也要ACK，避免pending堆积
                                redis_client.xack(current_stream_key, group_name, message_id)
                                continue
                            
                            # 推送给所有已连接的客户端（带去重）
                            for sid, client_info in list(clients.items()):
                                last_time = client_info.get('last_time', 0)
                                
                                if tick['time_msc'] > last_time:
                                    socketio.emit('tick_update', tick, room=sid)
                                    clients[sid]['last_time'] = tick['time_msc']
                            
                            # ACK确认消息（Redis Stream最佳实践）
                            # 🔴 修复：使用current_stream_key而不是stream_key，因为可能已切换到旧Stream
                            redis_client.xack(current_stream_key, group_name, message_id)
                            
                        except json.JSONDecodeError as e:
                            logger.error(f"JSON解析失败: {e}, message_id={message_id}")
                            # 🔴 修复：即使解析失败也要ACK，避免pending堆积
                            try:
                                redis_client.xack(current_stream_key, group_name, message_id)
                            except:
                                pass
                        except Exception as e:
                            logger.error(f"处理消息失败: {e}, message_id={message_id}")
                            # 🔴 修复：即使处理失败也要ACK，避免pending堆积
                            try:
                                redis_client.xack(current_stream_key, group_name, message_id)
                            except:
                                pass
            
            # 【最佳实践】定期推送K线数据（每60秒），但不阻塞主循环
            # 🔴 修复：使用try-except包裹，避免Redis操作阻塞事件循环
            current_time = time.time()
            if current_time - last_kline_push >= 60:
                try:
                    klines = get_kline_data_from_redis('BTCUSDm', '1m', 30)  # 修正：timeframe='1m', count=30
                    if klines:
                        socketio.emit('kline_update', klines)
                        last_kline_push = current_time
                except Exception as e:
                    logger.warning(f"推送K线数据失败: {e}")
                    # 即使失败也更新时间，避免频繁重试
                    last_kline_push = current_time
            
            # 【Flask-SocketIO最佳实践】使用 socketio.sleep() 让出控制权，避免阻塞事件循环
            # 不要使用 time.sleep()！
            socketio.sleep(0.001)  # 1ms，让事件循环有机会处理其他事件
                    
        except Exception as e:
            logger.error(f"Stream读取失败: {e}")
            socketio.sleep(1)  # 出错时等待1秒再重试


# ==================== MT5回调接口 ====================

@app.route('/api/mt5/callback', methods=['POST'])
def mt5_callback():
    """
    MT5中继服务回调接口
    
    接收MT5中继服务推送的实时更新：
    1. 新成交（deal）
    2. 持仓变化（positions_update）
    """
    try:
        data = request.json
        callback_type = data.get('type')
        
        if callback_type == 'deal':
            # 新成交，推送订单更新
            order_data = data.get('order')
            deal_data = data.get('deal', {})
            
            if order_data:
                logger.info(f"收到MT5成交回调: ticket={deal_data.get('ticket')}, order={deal_data.get('order')}")
                
                # 直接使用转换后的订单数据
                socketio.emit('order_update', order_data)
                logger.debug(f"推送订单更新: {order_data.get('orderId')}")
        
        elif callback_type == 'position_update':
            # 单个持仓更新（浮动盈亏变化）
            position = data.get('position')
            if position:
                socketio.emit('position_update', position)
                logger.debug(f"推送持仓更新: {position.get('positionId')}, 盈亏={position.get('unrealizedPnL', 0):.2f}")
        
        elif callback_type == 'positions_update':
            # 持仓变化，推送所有持仓更新
            positions = data.get('positions', [])
            logger.debug(f"收到MT5持仓更新回调: {len(positions)}个持仓")
            
            for pos in positions:
                frontend_position = format_position_for_frontend(pos)
                socketio.emit('position_update', frontend_position)
        
        return jsonify({'success': True, 'message': '回调处理成功'})
    
    except Exception as e:
        logger.error(f"处理MT5回调失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ==================== 布局配置管理 ====================

# 配置文件目录
CONFIG_DIR = BASE_DIR / "gui" / "user_configs"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

@app.route('/api/layout/save', methods=['POST'])
def save_layout():
    """保存用户布局配置"""
    try:
        data = request.json
        user_id = data.get('userId', 'default')
        page = data.get('page', 'trading')
        config = data.get('config', {})
        
        config_file = CONFIG_DIR / f'{user_id}_{page}_layout.json'
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✓ 布局配置已保存: {user_id}/{page}")
        return jsonify({'success': True, 'message': '布局保存成功'})
    
    except Exception as e:
        logger.error(f"保存布局配置失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/layout/load', methods=['GET'])
def load_layout():
    """加载用户布局配置"""
    try:
        user_id = request.args.get('userId', 'default')
        page = request.args.get('page', 'trading')
        
        config_file = CONFIG_DIR / f'{user_id}_{page}_layout.json'
        
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"✓ 布局配置已加载: {user_id}/{page}")
            return jsonify({'success': True, 'config': config})
        else:
            logger.info(f"未找到布局配置: {user_id}/{page}")
            return jsonify({'success': True, 'config': None})
    
    except Exception as e:
        logger.error(f"加载布局配置失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ==================== 订单管理API ====================

@app.route('/api/orders/create', methods=['POST'])
def create_order():
    """
    创建订单
    
    请求参数:
    {
        "symbol": "BTCUSDm",
        "action": "买入 (多)" | "卖出 (空)" | "平多" | "平空",
        "price": 100000.0,
        "klineTime": 1234567890,
        "volume": 0.01
    }
    """
    try:
        # 🔴 安全机制：检查是否为生产模式
        try:
            require_production_mode("创建订单")
        except EnvironmentError as e:
            logger.error(f"API订单创建被阻止: {str(e)}")
            return jsonify({
                'success': False,
                'error': 'ENVIRONMENT_BLOCKED',
                'message': f'非生产环境，订单创建已阻止。当前环境: {get_env_info().get("env", "UNKNOWN")}',
                'env_info': get_env_info()
            }), 403  # 403 Forbidden
        
        data = request.json
        
        symbol = data.get('symbol', 'BTCUSDm')
        action = data.get('action', '')
        price = float(data.get('price', 0))
        kline_time = int(data.get('klineTime', 0))
        volume = float(data.get('volume', 0.01))
        
        # 🔴 修复：支持标准英文参数（BUY/SELL），兼容中文参数（向后兼容）
        action_upper = action.upper()
        if action_upper == 'BUY' or '买入' in action or '做多' in action:
            order_type = OrderEngine.ORDER_TYPE_BUY
            comment = '买入开仓'
        elif action_upper == 'SELL' or '卖出' in action or '做空' in action:
            order_type = OrderEngine.ORDER_TYPE_SELL
            comment = '卖出开仓'
        elif '平多' in action:
            # 平多：需要找到对应的多单持仓并平仓
            positions = order_engine.get_all_positions()
            long_position = next((p for p in positions if p['type'] == OrderEngine.ORDER_TYPE_BUY), None)
            
            if long_position:
                close_order = order_engine.close_position(long_position['ticket'], price, kline_time)
                if close_order:
                    logger.info(f"✓ 平多成功: 票号={close_order['ticket']}, 价格={price}")
                    return jsonify({
                        'success': True,
                        'order': close_order,
                        'message': '平多成功'
                    })
                else:
                    return jsonify({'success': False, 'error': '平仓失败'}), 400
            else:
                return jsonify({'success': False, 'error': '没有多单持仓'}), 400
                
        elif '平空' in action:
            # 平空：需要找到对应的空单持仓并平仓
            positions = order_engine.get_all_positions()
            short_position = next((p for p in positions if p['type'] == OrderEngine.ORDER_TYPE_SELL), None)
            
            if short_position:
                close_order = order_engine.close_position(short_position['ticket'], price, kline_time)
                if close_order:
                    logger.info(f"✓ 平空成功: 票号={close_order['ticket']}, 价格={price}")
                    return jsonify({
                        'success': True,
                        'order': close_order,
                        'message': '平空成功'
                    })
                else:
                    return jsonify({'success': False, 'error': '平仓失败'}), 400
            else:
                return jsonify({'success': False, 'error': '没有空单持仓'}), 400
        else:
            return jsonify({'success': False, 'error': '未知操作类型'}), 400
        
        # 🚀 推送到MT5执行（通过Redis Stream）
        # 写入 l3:manual:commands Stream，由 OrderExecutor 监听并执行
        manual_command = {
            'action': 'BUY' if order_type == OrderEngine.ORDER_TYPE_BUY else 'SELL',
            'symbol': symbol,
            'price': price if price > 0 else 0.0,  # 市价单为0，MT5会使用当前市价
            'volume': volume,
            'sl': 0.0,  # 止损（当前未实现）
            'tp': 0.0,  # 止盈（当前未实现）
            'source': 'MANUAL',  # 标记为人工订单
            'timestamp': int(time.time()),
            'klineTime': kline_time,
        }
        
        # 写入Redis Stream
        try:
            redis_client.xadd('l3:manual:commands', manual_command, maxlen=1000)
            logger.info(f"✓ 订单已推送到MT5执行队列: {manual_command['action']} {symbol} {volume}手 @ {price}")
        except Exception as e:
            logger.error(f"推送订单到MT5队列失败: {e}")
            return jsonify({'success': False, 'error': f'订单推送失败: {str(e)}'}), 500
        
        # 返回响应（订单将在MT5执行后通过Socket.IO推送更新）
        return jsonify({
            'success': True,
            'message': '订单已提交到MT5执行队列',
            'command': manual_command
        })
    
    except Exception as e:
        logger.error(f"创建订单失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/orders', methods=['GET'])
@app.route('/api/orders/open', methods=['GET'])  # 🔴 修复：添加别名，兼容前端调用
def get_orders():
    """获取当前挂单（Pending Orders）"""
    try:
        # 🚀 快速响应：如果order_engine不可用，直接返回空列表
        if order_engine is None:
            return jsonify({
                'success': True,
                'orders': [],
                'count': 0
            })
        
        # 🔴 修复：添加超时保护，避免gRPC调用阻塞
        import concurrent.futures
        all_orders = []
        positions = []
        
        try:
            # 使用线程池执行器，设置超时保护（2秒）
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                # 提交任务
                future_orders = executor.submit(order_engine.get_all_orders)
                future_positions = executor.submit(order_engine.get_all_positions)
                
                # 等待结果，设置超时（2秒）
                try:
                    all_orders = future_orders.result(timeout=2.0)
                    positions = future_positions.result(timeout=2.0)
                except concurrent.futures.TimeoutError:
                    logger.warning(f"获取订单/持仓超时（2秒），返回空列表")
                    all_orders = []
                    positions = []
        except Exception as e:
            logger.warning(f"获取订单/持仓失败（gRPC可能超时）: {e}")
            all_orders = []
            positions = []  # 返回空列表，不阻塞API
        
        # 获取当前持仓的position_id列表
        position_ids = {pos['ticket'] for pos in positions}
        
        # 筛选挂单：state=1（已下单但未成交）且不在持仓中
        pending_orders = [
            order for order in all_orders 
            if order.get('state') == 1 and order.get('position_id', 0) not in position_ids
        ]
        
        return jsonify({
            'success': True,
            'orders': pending_orders,
            'count': len(pending_orders)
        })
                
    except Exception as e:
        logger.error(f"获取挂单失败: {e}")
        # 🚀 降级：返回成功但空列表，避免前端显示错误
        return jsonify({
            'success': True,
            'orders': [],
            'count': 0
        })


@app.route('/api/orders/positions', methods=['GET'])
def get_positions():
    """
    获取所有持仓（统一返回前端格式）
    
    使用统一的 format_position_for_frontend 函数确保与 WebSocket 推送格式一致。
    """
    try:
        # 🚀 快速响应：如果order_engine不可用，直接返回空列表
        if order_engine is None:
            return jsonify({
                'success': True,
                'data': []
            })
        
        # 🔴 修复：添加超时保护，避免gRPC调用阻塞
        import concurrent.futures
        positions = []
        
        try:
            # 使用线程池执行器，设置超时保护（2秒）
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                # 提交任务
                future = executor.submit(order_engine.get_all_positions)
                
                # 等待结果，设置超时（2秒）
                try:
                    positions = future.result(timeout=2.0)
                except concurrent.futures.TimeoutError:
                    logger.warning(f"获取持仓超时（2秒），返回空列表")
                    positions = []
        except Exception as e:
            logger.warning(f"获取持仓失败（gRPC可能超时）: {e}")
            positions = []  # 返回空列表，不阻塞API
        
        # 🔴 统一转换为前端格式（与WebSocket推送格式一致）
        frontend_positions = [format_position_for_frontend(pos) for pos in positions]
        
        logger.debug(f"📤 HTTP API 返回持仓 {len(frontend_positions)} 个 (已转换为前端格式)")
        
        return jsonify({
            'success': True,
            'data': frontend_positions
        })
                
    except Exception as e:
        logger.error(f"❌ 获取持仓失败: {e}")
        # 🚀 降级：返回成功但空列表，避免前端显示错误
        return jsonify({
            'success': True,
            'data': []
        })


@app.route('/api/orders/history', methods=['GET'])
def get_order_history():
    """
    获取历史订单（只返回已平仓的订单）
    
    🚀 优化：快速响应，优先使用Redis缓存，gRPC调用带超时保护
    """
    try:
        # 🚀 快速响应：如果order_engine不可用，直接返回空列表
        if order_engine is None:
            return jsonify({
                'success': True,
                'data': []
            })
        
        # 优先从Redis缓存读取，避免gRPC阻塞
        all_orders = []
        positions = []
        
        # 方法1: 尝试从Redis缓存快速读取（O(1)操作）
        try:
            # 直接调用内部方法，避免gRPC调用
            all_orders = order_engine._get_orders_from_redis_cache()
            positions = order_engine._get_positions_from_redis_cache()
            logger.debug(f"从Redis缓存读取: {len(all_orders)} 订单, {len(positions)} 持仓")
        except Exception as e:
            logger.debug(f"Redis缓存读取失败: {e}，尝试gRPC")
        
        # 方法2: 如果缓存为空，尝试快速gRPC调用（带超时，最多1秒）
        if not all_orders or not positions:
            try:
                # 🚀 使用超时保护，避免阻塞（最多1秒）
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                    future_orders = executor.submit(order_engine.get_all_orders)
                    future_positions = executor.submit(order_engine.get_all_positions)
                    
                    try:
                        all_orders = future_orders.result(timeout=1.0) or all_orders  # 减少到1秒
                        positions = future_positions.result(timeout=1.0) or positions
                    except concurrent.futures.TimeoutError:
                        logger.warning("gRPC调用超时（1秒），使用Redis缓存数据或空列表")
                        # 如果超时，使用已有数据或空列表
                        all_orders = all_orders or []
                        positions = positions or []
            except Exception as e:
                logger.warning(f"gRPC调用失败: {e}，使用Redis缓存数据或空列表")
                # 如果失败，使用已有数据或空列表
                all_orders = all_orders or []
                positions = positions or []
        
        # 获取当前持仓的position_id列表（安全处理）
        position_ids = set()
        for pos in positions:
            try:
                ticket = pos.get('ticket') or pos.get('position_id')
                if ticket:
                    position_ids.add(ticket)
            except Exception:
                continue
        
        # 只返回不在持仓中的订单（已平仓的）
        history_orders = []
        for order in all_orders:
            try:
                order_position_id = order.get('position_id') or order.get('ticket')
                if order_position_id and order_position_id not in position_ids:
                    history_orders.append(order)
            except Exception as e:
                logger.debug(f"处理订单时出错: {e}，跳过该订单")
                continue
        
        return jsonify({
            'success': True,
            'data': history_orders
        })
    except Exception as e:
        logger.error(f"获取历史订单失败: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/orders/deals', methods=['GET'])
def get_deals():
    """获取成交记录"""
    try:
        deals = order_engine.get_all_deals()
        return jsonify({
            'success': True,
            'data': deals
        })
    except Exception as e:
        logger.error(f"获取成交记录失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/orders/close', methods=['POST'])
def close_position():
    """
    平仓
    
    🚀 优化：添加超时保护，避免gRPC调用阻塞
    """
    try:
        # 🚀 快速响应：如果order_engine不可用，直接返回错误
        if order_engine is None:
            return jsonify({
                'success': False,
                'error': '订单引擎不可用'
            }), 503
        
        data = request.json
        position_id = data.get('position_id')
        close_price = data.get('close_price')
        kline_time = data.get('kline_time')
        
        # 参数验证：position_id 是必需的，close_price 和 kline_time 可选（后端会使用当前值）
        if not position_id:
            return jsonify({
                'success': False,
                'error': '缺少必要参数: position_id'
            }), 400
        
        # 🚀 使用超时保护，避免gRPC调用阻塞（最多3秒）
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                order_engine.close_position,
                position_id=int(position_id),
                close_price=float(close_price) if close_price else 0,
                kline_time=int(kline_time) if kline_time else 0
            )
            
            try:
                closed_order = future.result(timeout=3.0)  # 3秒超时
            except concurrent.futures.TimeoutError:
                logger.warning(f"平仓操作超时（3秒）: position_id={position_id}")
                return jsonify({
                    'success': False,
                    'error': '平仓操作超时，请稍后重试'
                }), 504
        
        if closed_order:
            logger.info(f"✓ 平仓成功: position_id={position_id}, price={close_price}")
            return jsonify({
                'success': True,
                'order': closed_order,
                'message': '平仓成功'
            })
        else:
            return jsonify({
                'success': False,
                'error': '持仓不存在或平仓失败'
            }), 404
            
    except Exception as e:
        logger.error(f"平仓失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/orders/clear', methods=['DELETE'])
def clear_orders():
    """清除所有订单数据"""
    try:
        order_engine.clear_all_orders()
        logger.info("已清除所有订单数据")
        return jsonify({
            'success': True,
            'message': '已清除所有订单数据'
        })
    except Exception as e:
        logger.error(f"清除订单数据失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ==================== 训练配置API ====================

@app.route('/api/training/config', methods=['GET', 'POST'])
def training_config():
    """训练配置管理"""
    try:
        if request.method == 'GET':
            # 获取训练配置
            config_data = redis_client.get('training:config')
            if config_data:
                config = json.loads(config_data)
                logger.info("✓ 读取训练配置")
                return jsonify({
                    'success': True,
                    'time_windows': config.get('time_windows', []),
                    'primary_window': config.get('primary_window', 'medium'),
                    'use_open_price': config.get('use_open_price', True)
                })
            else:
                # 返回默认配置
                from config.model_config import TIME_WINDOWS, DATA_PREPARATION_CONFIG
                logger.info("⚠️  训练配置不存在，返回默认配置")
                return jsonify({
                    'success': True,
                    'time_windows': TIME_WINDOWS,
                    'primary_window': DATA_PREPARATION_CONFIG['primary_window'],
                    'use_open_price': DATA_PREPARATION_CONFIG['use_open_price']
                })
        
        elif request.method == 'POST':
            # 保存训练配置
            config_data = request.json
            redis_client.set('training:config', json.dumps(config_data))
            logger.info(f"✓ 保存训练配置: {len(config_data.get('time_windows', []))} 个时间窗口")
            return jsonify({
                'success': True,
                'message': '训练配置已保存'
            })
    
    except Exception as e:
        logger.error(f"训练配置API失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/training/images/<path:filename>')
def serve_training_image(filename):
    """提供训练数据图片访问"""
    try:
        image_dir = BASE_DIR / 'data' / 'images'
        return send_from_directory(image_dir, filename)
    except Exception as e:
        logger.error(f"图片访问失败: {e}")
        return jsonify({'error': str(e)}), 404


# ==================== AI分身配置API ====================

@app.route('/api/ai-avatars', methods=['GET'])
def get_ai_avatars():
    """获取所有AI分身配置"""
    try:
        from config.ai_avatars import get_all_avatars
        
        avatars = get_all_avatars()
        logger.info(f"✓ 返回 {len(avatars)} 个AI分身配置")
        
        return jsonify({
            'success': True,
            'avatars': avatars
        })
    except Exception as e:
        logger.error(f"获取AI分身配置失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/ai-avatars/save', methods=['POST'])
def save_ai_avatar():
    """保存AI分身配置"""
    try:
        data = request.get_json()
        avatar = data.get('avatar')
        config = data.get('config')
        
        if not avatar or not config:
            return jsonify({'success': False, 'error': '缺少必要参数'}), 400
        
        # 保存到Redis
        avatar_key = f"ai_avatar:{avatar['id']}"
        avatar_data = {
            'id': avatar['id'],
            'name': avatar['name'],
            'description': avatar['description'],
            'icon': avatar['icon'],
            'color': avatar['color'],
            'config': json.dumps(config)
        }
        
        redis_client.hset(avatar_key, mapping=avatar_data)
        logger.info(f"✓ AI分身配置已保存: {avatar['name']}")
        
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"保存AI分身配置失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/ai-avatars/<avatar_id>/train', methods=['POST'])
def start_ai_training(avatar_id):
    """启动AI分身训练"""
    try:
        # 获取AI分身配置
        avatar_key = f"ai_avatar:{avatar_id}"
        avatar_data = redis_client.hgetall(avatar_key)
        
        if not avatar_data:
            return jsonify({'success': False, 'error': 'AI分身不存在'}), 404
        
        # 更新训练状态
        training_key = f"ai_training:{avatar_id}"
        training_data = {
            'status': 'training',
            'start_time': str(int(time.time())),
            'progress': 0
        }
        redis_client.hset(training_key, mapping=training_data)
        
        # 启动异步训练
        from training.ai_avatar_trainer import train_avatar_async
        train_avatar_async(avatar_id)
        
        logger.info(f"✓ AI分身 {avatar_id} 训练已启动")
        
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"启动AI训练失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/ai-avatars/<avatar_id>/stop', methods=['POST'])
def stop_ai_training(avatar_id):
    """停止AI分身训练"""
    try:
        training_key = f"ai_training:{avatar_id}"
        redis_client.hset(training_key, 'status', 'stopped')
        
        logger.info(f"✓ AI分身 {avatar_id} 训练已停止")
        
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"停止AI训练失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/ai-avatars/battle-stats', methods=['GET'])
def get_battle_stats():
    """获取AI分身对战统计"""
    try:
        stats = {}
        
        # 从Redis获取所有AI分身的统计信息
        avatar_keys = redis_client.keys("ai_avatar:*")
        
        for avatar_key in avatar_keys:
            avatar_id = avatar_key.split(':')[1]
            stats_key = f"ai_stats:{avatar_id}"
            
            # 获取统计数据
            stats_data = redis_client.hgetall(stats_key)
            if stats_data:
                # 转换数据类型
                stats[avatar_id] = {
                    'avatar_id': avatar_id,
                    'total_trades': int(stats_data.get('total_trades', 0)),
                    'win_trades': int(stats_data.get('win_trades', 0)),
                    'lose_trades': int(stats_data.get('lose_trades', 0)),
                    'win_rate': float(stats_data.get('win_rate', 0)),
                    'total_profit': float(stats_data.get('total_profit', 0)),
                    'avg_profit_per_trade': float(stats_data.get('avg_profit_per_trade', 0)),
                    'max_profit': float(stats_data.get('max_profit', 0)),
                    'max_loss': float(stats_data.get('max_loss', 0)),
                    'sharpe_ratio': float(stats_data.get('sharpe_ratio', 0)),
                    'max_drawdown': float(stats_data.get('max_drawdown', 0)),
                    'profit_factor': float(stats_data.get('profit_factor', 0)),
                    'training_samples': int(stats_data.get('training_samples', 0)),
                    'model_accuracy': float(stats_data.get('model_accuracy', 0)),
                    'training_time': float(stats_data.get('training_time', 0)),
                    'last_updated': stats_data.get('last_updated')
                }
            else:
                # 返回默认统计
                stats[avatar_id] = {
                    'avatar_id': avatar_id,
                    'total_trades': 0,
                    'win_trades': 0,
                    'lose_trades': 0,
                    'win_rate': 0,
                    'total_profit': 0,
                    'avg_profit_per_trade': 0,
                    'max_profit': 0,
                    'max_loss': 0,
                    'sharpe_ratio': 0,
                    'max_drawdown': 0,
                    'profit_factor': 0,
                    'training_samples': 0,
                    'model_accuracy': 0,
                    'training_time': 0,
                    'last_updated': None
                }
        
        logger.info(f"✓ 返回 {len(stats)} 个AI分身的对战统计")
        
        return jsonify({
            'success': True,
            'stats': stats
        })
    except Exception as e:
        logger.error(f"获取对战统计失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/training/generate', methods=['POST'])
def generate_training_data():
    """生成训练数据（直接在Flask进程内执行，避免subprocess阻塞）"""
    try:
        import importlib.util
        
        logger.info("开始生成训练数据（Flask进程内执行）")
        
        # 🔥 直接加载并执行prepare_data.py模块
        prepare_data_path = BASE_DIR / 'training' / 'prepare_data.py'
        
        spec = importlib.util.spec_from_file_location("prepare_data", prepare_data_path)
        prepare_data_module = importlib.util.module_from_spec(spec)
        
        # 执行模块（会自动调用main()）
        spec.loader.exec_module(prepare_data_module)
        
        # 读取生成的元数据
        metadata_path = BASE_DIR / 'data' / 'training_metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                training_samples = json.load(f)
            
            logger.info(f"[OK] 生成训练数据成功: {len(training_samples)} 个样本")
            
            return jsonify({
                'success': True,
                'samples': training_samples,
                'message': f'成功生成 {len(training_samples)} 个训练样本'
            })
        else:
            logger.error("元数据文件不存在")
            return jsonify({
                'success': False,
                'error': '训练数据未生成'
            }), 500
    
    except Exception as e:
        logger.error(f"生成训练数据异常: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


# ==================== MT5 API ====================

# ❌ 已删除：所有/api/mt5/*相关的API端点（共7个）
# 现在使用：数据采集器 + Redis + WebSocket

# ==================== Redis数据API ====================

@app.route('/api/redis/tick/<symbol>', methods=['GET'])
def get_tick_from_redis(symbol):
    """从Redis获取TICK数据"""
    try:
        # 从Redis获取最新的TICK数据
        tick_key = f"tick:{symbol.upper()}"
        tick_data = redis_client.hgetall(tick_key)
        
        if not tick_data:
            return jsonify({
                'success': False,
                'error': f'未找到 {symbol} 的TICK数据'
            }), 404
        
        # 转换数据类型
        tick = {
            'symbol': tick_data.get('symbol', symbol.upper()),
            'time': tick_data.get('time', ''),
            'bid': float(tick_data.get('bid', 0)),
            'ask': float(tick_data.get('ask', 0)),
            'last': float(tick_data.get('last', 0)),
            'volume': int(tick_data.get('volume', 0)),
            'spread': float(tick_data.get('spread', 0)),
            'change': float(tick_data.get('change', 0)),
            'change_percent': float(tick_data.get('change_percent', 0))
        }
        
        return jsonify({
            'success': True,
            'tick': tick
        })
        
    except Exception as e:
        logger.error(f"从Redis获取TICK数据失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/redis/ticks/history/<symbol>', methods=['GET'])
def api_get_tick_history_from_redis(symbol):
    """API: 从Redis获取TICK历史数据"""
    try:
        # 获取参数
        limit = request.args.get('limit', 100, type=int)
        
        # 从Redis获取历史TICK数据
        history_key = f"tick_history:{symbol.upper()}"
        tick_list = redis_client.lrange(history_key, 0, limit - 1)
        
        ticks = []
        for tick_json in tick_list:
            try:
                tick_data = json.loads(tick_json)
                ticks.append(tick_data)
            except json.JSONDecodeError:
                continue
        
        return jsonify({
            'success': True,
            'ticks': ticks,
            'count': len(ticks)
        })
        
    except Exception as e:
        logger.error(f"从Redis获取TICK历史数据失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/assets', methods=['GET'])
def get_assets():
    """
    获取所有可交易资产列表
    
    🚀 优化：快速响应，避免使用KEYS命令（阻塞操作）
    
    返回格式：
    [
        {
            "symbol": "BTCUSDm",
            "name": "Bitcoin/USD",
            "baseCurrency": "BTC",
            "quoteCurrency": "USD",
            "minVolume": 0.01,
            "maxVolume": 100.0,
            "pricePrecision": 2,
            "volumePrecision": 2,
            "status": "ACTIVE"
        }
    ]
    """
    try:
        # 🚀 优化：使用SCAN替代KEYS，避免阻塞（但为了快速响应，直接返回默认列表）
        # KEYS命令会阻塞Redis，导致API超时
        # 直接返回默认资产列表，或从配置中读取
        symbols = {'BTCUSDm'}  # 默认资产
        
        # 🔴 修复：移除Redis Stream检查，直接返回默认列表，避免阻塞
        # xinfo_stream可能在某些情况下阻塞，导致API超时
        # 直接使用默认资产列表，确保快速响应
        
        # 构建资产列表
        assets = []
        for symbol in sorted(symbols):
            # 解析symbol，提取基础货币和报价货币
            base_currency = symbol.replace('USDm', '').replace('USD', '').replace('XAU', 'GOLD')
            quote_currency = 'USD'
            
            # 根据symbol类型设置精度
            if 'XAU' in symbol or 'GOLD' in symbol:
                price_precision = 2
            elif 'USD' in symbol:
                price_precision = 2
            else:
                price_precision = 5
            
            asset = {
                'symbol': symbol,
                'name': f'{base_currency}/{quote_currency}',
                'baseCurrency': base_currency,
                'quoteCurrency': quote_currency,
                'minVolume': 0.01,
                'maxVolume': 100.0,
                'pricePrecision': price_precision,
                'volumePrecision': 2,
                'status': 'ACTIVE'
            }
            assets.append(asset)
        
        logger.info(f"✓ 返回 {len(assets)} 个可交易资产")
        # 返回统一格式：{success: true, data: [...]}
        return jsonify({
            'success': True,
            'data': assets
        })
        
    except Exception as e:
        logger.error(f"获取资产列表失败: {e}")
        # 返回默认资产列表（至少保证系统可用）
        default_assets = [{
            'symbol': 'BTCUSDm',
            'name': 'Bitcoin/USD',
            'baseCurrency': 'BTC',
            'quoteCurrency': 'USD',
            'minVolume': 0.01,
            'maxVolume': 100.0,
            'pricePrecision': 2,
            'volumePrecision': 2,
            'status': 'ACTIVE'
        }]
        return jsonify({
            'success': True,
            'data': default_assets
        })


@app.route('/api/account', methods=['GET'])
@app.route('/api/account/info', methods=['GET'])  # 🔴 修复：添加别名，兼容前端调用
def get_account_info():
    """获取账户信息（管理信息）- 混合架构：同步查询（gRPC）"""
    try:
        from src.trading.services.grpc_trade_client import get_grpc_client, is_grpc_available
        
        if not is_grpc_available():
            return jsonify({
                'success': False,
                'error': 'gRPC 功能不可用'
            }), 503
        
        # 🚀 添加超时保护（2秒），避免阻塞API
        import concurrent.futures
        try:
            client = get_grpc_client()
            
            # 使用线程池执行器，设置超时保护（2秒）
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                # 提交任务
                future = executor.submit(client.get_account_info)
            
                # 等待结果，设置超时（2秒）
                try:
                    result = future.result(timeout=2.0)
                except concurrent.futures.TimeoutError:
                    logger.warning(f"获取账户信息超时（2秒）")
                    return jsonify({
                        'success': False,
                        'error': '请求超时，请稍后重试'
                    }), 504
            
            if result.get('success'):
                return jsonify({
                    'success': True,
                    'data': result.get('account_info')
                })
            else:
                return jsonify({
                    'success': False,
                    'error': result.get('message', '获取账户信息失败')
                }), 500
        except Exception as grpc_error:
            # 捕获所有gRPC相关错误
            error_msg = str(grpc_error)
            logger.warning(f"获取账户信息失败: {error_msg}")
            return jsonify({
                'success': False,
                'error': '获取账户信息失败，请稍后重试'
            }), 500
            
    except Exception as e:
        logger.error(f"获取账户信息失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/symbols/<symbol>', methods=['GET'])
def get_symbol_info(symbol):
    """获取品种信息（管理信息）- 混合架构：同步查询（gRPC）"""
    try:
        from src.trading.services.grpc_trade_client import get_grpc_client, is_grpc_available
        import concurrent.futures
        
        if not is_grpc_available():
            return jsonify({
                'success': False,
                'error': 'gRPC 功能不可用'
            }), 503
        
        # 🔴 修复：添加超时保护，避免阻塞API
        try:
            client = get_grpc_client()
            
            # 使用线程池执行器，设置超时保护（2秒）
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(client.get_symbol_info, symbol)
                try:
                    result = future.result(timeout=2.0)
                except concurrent.futures.TimeoutError:
                    logger.warning(f"获取品种信息超时（2秒）")
                    return jsonify({
                        'success': False,
                        'error': '请求超时，请稍后重试'
                    }), 504
            
            if result.get('success'):
                return jsonify({
                    'success': True,
                    'data': result.get('symbol_info')
                })
            else:
                return jsonify({
                    'success': False,
                    'error': result.get('message', '获取品种信息失败')
                }), 500
        except Exception as grpc_error:
            logger.error(f"gRPC调用失败: {grpc_error}")
            return jsonify({
                'success': False,
                'error': f'gRPC调用失败: {str(grpc_error)[:100]}'
            }), 500
            
    except Exception as e:
        logger.error(f"获取品种信息失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/terminal', methods=['GET'])
def get_terminal_info():
    """获取终端信息（管理信息）- 混合架构：同步查询（gRPC）"""
    try:
        from src.trading.services.grpc_trade_client import get_grpc_client, is_grpc_available
        import concurrent.futures
        
        if not is_grpc_available():
            return jsonify({
                'success': False,
                'error': 'gRPC 功能不可用'
            }), 503
        
        # 🔴 修复：添加超时保护，避免阻塞API
        try:
            client = get_grpc_client()
            
            # 使用线程池执行器，设置超时保护（2秒）
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(client.get_terminal_info)
                try:
                    result = future.result(timeout=2.0)
                except concurrent.futures.TimeoutError:
                    logger.warning(f"获取终端信息超时（2秒）")
                    return jsonify({
                        'success': False,
                        'error': '请求超时，请稍后重试'
                    }), 504
            
            if result.get('success'):
                return jsonify({
                    'success': True,
                    'data': result.get('terminal_info')
                })
            else:
                return jsonify({
                    'success': False,
                    'error': result.get('message', '获取终端信息失败')
                }), 500
        except Exception as grpc_error:
            logger.error(f"gRPC调用失败: {grpc_error}")
            return jsonify({
                'success': False,
                'error': f'gRPC调用失败: {str(grpc_error)[:100]}'
            }), 500
            
    except Exception as e:
        logger.error(f"获取终端信息失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/server/time', methods=['GET'])
def get_server_time():
    """获取服务器时间（管理信息）- 混合架构：同步查询（gRPC）"""
    try:
        from src.trading.services.grpc_trade_client import get_grpc_client, is_grpc_available
        import concurrent.futures
        
        if not is_grpc_available():
            return jsonify({
                'success': False,
                'error': 'gRPC 功能不可用'
            }), 503
        
        # 🔴 修复：添加超时保护，避免阻塞API
        try:
            client = get_grpc_client()
            
            # 使用线程池执行器，设置超时保护（2秒）
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(client.get_server_time)
                try:
                    result = future.result(timeout=2.0)
                except concurrent.futures.TimeoutError:
                    logger.warning(f"获取服务器时间超时（2秒）")
                    return jsonify({
                        'success': False,
                        'error': '请求超时，请稍后重试'
                    }), 504
            
            if result.get('success'):
                return jsonify({
                    'success': True,
                    'data': {
                        'time': result.get('time'),
                        'time_msc': result.get('time_msc')
                    }
                })
            else:
                return jsonify({
                    'success': False,
                    'error': result.get('message', '获取服务器时间失败')
                }), 500
        except Exception as grpc_error:
            logger.error(f"gRPC调用失败: {grpc_error}")
            return jsonify({
                'success': False,
                'error': f'gRPC调用失败: {str(grpc_error)[:100]}'
            }), 500
            
    except Exception as e:
        logger.error(f"获取服务器时间失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/symbols', methods=['GET'])
def get_symbol_list():
    """获取品种列表（管理信息）- 混合架构：同步查询（gRPC）"""
    try:
        from src.trading.services.grpc_trade_client import get_grpc_client, is_grpc_available
        import concurrent.futures
        
        group = request.args.get('group', '')
        
        if not is_grpc_available():
            return jsonify({
                'success': False,
                'error': 'gRPC 功能不可用'
            }), 503
        
        # 🔴 修复：添加超时保护，避免阻塞API
        try:
            client = get_grpc_client()
            
            # 使用线程池执行器，设置超时保护（2秒）
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(client.get_symbol_list, group=group)
                try:
                    result = future.result(timeout=2.0)
                except concurrent.futures.TimeoutError:
                    logger.warning(f"获取品种列表超时（2秒）")
                    return jsonify({
                        'success': False,
                        'error': '请求超时，请稍后重试'
                    }), 504
            
            if result.get('success'):
                return jsonify({
                    'success': True,
                    'data': {
                        'symbols': result.get('symbols', []),
                        'count': result.get('count', 0)
                    }
                })
            else:
                return jsonify({
                    'success': False,
                    'error': result.get('message', '获取品种列表失败')
                }), 500
        except Exception as grpc_error:
            logger.error(f"gRPC调用失败: {grpc_error}")
            return jsonify({
                'success': False,
                'error': f'gRPC调用失败: {str(grpc_error)[:100]}'
            }), 500
            
    except Exception as e:
        logger.error(f"获取品种列表失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/redis/symbols', methods=['GET'])
def get_available_symbols():
    """获取Redis中可用的交易品种"""
    try:
        # 从Redis获取所有TICK数据的key
        tick_keys = redis_client.keys("tick:*")
        symbols = []
        
        for key in tick_keys:
            symbol = key.decode('utf-8').replace('tick:', '')
            symbols.append(symbol)
        
        return jsonify({
            'success': True,
            'symbols': symbols
        })
        
    except Exception as e:
        logger.error(f"获取可用品种失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/redis/latest-tick', methods=['GET'])
def get_latest_tick_from_collector():
    """从数据采集器获取最新TICK数据"""
    try:
        # 从数据采集器的Redis键获取最新TICK数据
        latest_tick_key = 'tick:BTCUSDm:latest'
        tick_data = redis_client.get(latest_tick_key)
        
        if not tick_data:
            return jsonify({
                'success': False,
                'error': '未找到最新TICK数据，请确保数据采集器正在运行'
            }), 404
        
        # 解析JSON数据
        tick = json.loads(tick_data)
        
        return jsonify({
            'success': True,
            'tick': tick
        })
        
    except Exception as e:
        logger.error(f"获取最新TICK数据失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/redis/tick-history', methods=['GET'])
def get_tick_history_from_collector():
    """从数据采集器获取TICK历史数据"""
    try:
        # 获取参数
        limit = request.args.get('limit', 100, type=int)
        
        # 从数据采集器的Redis键获取TICK历史数据
        tick_history_key = 'tick:BTCUSDm:realtime'
        tick_list = redis_client.zrange(tick_history_key, -limit, -1)  # 获取最新的N条
        
        ticks = []
        for tick_json in tick_list:
            try:
                tick_data = json.loads(tick_json)
                ticks.append(tick_data)
            except json.JSONDecodeError:
                continue
        
        return jsonify({
            'success': True,
            'ticks': ticks,
            'count': len(ticks)
        })
        
    except Exception as e:
        logger.error(f"获取TICK历史数据失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/strategy/logs', methods=['GET'])
def get_strategy_logs():
    """获取策略日志/信号（从Redis Stream读取）
    
    参数:
        symbol: 交易对 (默认: BTCUSDm)
        count: 获取数量 (默认: 100)
    """
    try:
        symbol = request.args.get('symbol', 'BTCUSDm')
        count = request.args.get('count', 100, type=int)
        
        # 从Redis Stream读取策略信号
        signal_stream_key = f"signal:{symbol}:stream"
        signal_history_key = f"signal:{symbol}:history"
        
        logs = []
        
        # 优先从Sorted Set读取（历史信号，按时间排序）
        try:
            # 获取最近的count条信号
            signal_jsons = redis_client.zrange(signal_history_key, -count, -1, withscores=False)
            
            for signal_json in signal_jsons:
                try:
                    signal = json.loads(signal_json)
                    # 转换为前端格式
                    action = signal.get('action', 'UNKNOWN')
                    reason = signal.get('reason', '')
                    price = signal.get('price', 0)
                    
                    # 确定日志级别
                    if action in ['BUY', 'SELL']:
                        level = 'SIGNAL'
                        message = f"{symbol} {action} 信号 @ {price:.2f}"
                        details = reason
                    else:
                        level = 'INFO'
                        message = f"{symbol} 状态更新: {action}"
                        details = reason
                    
                    logs.append({
                        'timestamp': int(signal.get('timestamp', signal.get('tick_time_ms', 0) / 1000)),
                        'level': level,
                        'message': message,
                        'details': details,
                        'signal': signal  # 保留完整信号数据
                    })
                except Exception as e:
                    logger.warning(f"解析策略信号失败: {e}")
                    continue
        except Exception as e:
            logger.warning(f"从Redis读取策略信号失败: {e}")
        
        # 如果没有从Sorted Set获取到数据，尝试从Stream读取
        if not logs:
            try:
                # 从Stream读取最新消息
                messages = redis_client.xrevrange(signal_stream_key, count=count)
                for msg_id, fields in messages:
                    signal_json = fields.get('signal_json')
                    if signal_json:
                        try:
                            signal = json.loads(signal_json)
                            action = signal.get('action', 'UNKNOWN')
                            reason = signal.get('reason', '')
                            price = signal.get('price', 0)
                            
                            if action in ['BUY', 'SELL']:
                                level = 'SIGNAL'
                                message = f"{symbol} {action} 信号 @ {price:.2f}"
                            else:
                                level = 'INFO'
                                message = f"{symbol} 状态更新: {action}"
                            
                            logs.append({
                                'timestamp': int(signal.get('timestamp', signal.get('tick_time_ms', 0) / 1000)),
                                'level': level,
                                'message': message,
                                'details': reason,
                                'signal': signal
                            })
                        except Exception as e:
                            logger.warning(f"解析Stream信号失败: {e}")
                            continue
            except Exception as e:
                logger.warning(f"从Redis Stream读取策略信号失败: {e}")
        
        # 按时间戳降序排序（最新的在前）
        logs.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return jsonify({
            'success': True,
            'data': logs,
            'count': len(logs)
        })
    except Exception as e:
        logger.error(f"获取策略日志失败: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'data': [],
            'error': str(e)
        }), 500


@app.route('/api/redis/collector-status', methods=['GET'])
def get_collector_status():
    """获取数据采集器状态"""
    try:
        # 从数据采集器的状态键获取状态信息
        status_key = 'status:BTCUSDm:collector'
        status_data = redis_client.get(status_key)
        
        if not status_data:
            return jsonify({
                'success': False,
                'error': '未找到数据采集器状态，请确保数据采集器正在运行'
            }), 404
        
        # 解析JSON数据
        status = json.loads(status_data)
        
        return jsonify({
            'success': True,
            'status': status
        })
        
    except Exception as e:
        logger.error(f"获取数据采集器状态失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/system/services', methods=['GET'])
def get_system_services():
    """获取系统服务状态（systemd）"""
    import subprocess
    import json as json_lib
    
    services = {
        'mt5-collector': {
            'name': 'MT5数据采集器',
            'description': '从MetaTrader 5采集市场数据',
            'service': 'mt5-collector.service'
        },
        'hft-core': {
            'name': 'HFT核心服务',
            'description': '策略状态机、交易执行器、监控服务',
            'service': 'hft-core.service'
        },
        'backend-api': {
            'name': '后端API服务',
            'description': 'Flask API和WebSocket服务',
            'service': 'backend-api.service'
        }
    }
    
    result = {}
    
    for key, info in services.items():
        service_name = info['service']
        try:
            # 检查服务状态
            status_cmd = ['systemctl', 'is-active', service_name]
            is_active = subprocess.run(status_cmd, capture_output=True, text=True, timeout=2)
            active = is_active.stdout.strip() == 'active'
            
            # 获取服务详细信息
            show_cmd = ['systemctl', 'show', service_name, '--property=ActiveState,SubState,MainPID,LoadState']
            show_result = subprocess.run(show_cmd, capture_output=True, text=True, timeout=2)
            
            # 解析输出
            service_info = {}
            for line in show_result.stdout.strip().split('\n'):
                if '=' in line:
                    k, v = line.split('=', 1)
                    service_info[k] = v
            
            # 获取日志（最近几行）
            try:
                journal_cmd = ['journalctl', '-u', service_name, '-n', '5', '--no-pager', '-o', 'json']
                journal_result = subprocess.run(journal_cmd, capture_output=True, text=True, timeout=2)
                logs = []
                for line in journal_result.stdout.strip().split('\n'):
                    if line:
                        try:
                            log_entry = json_lib.loads(line)
                            logs.append({
                                'message': log_entry.get('MESSAGE', ''),
                                'timestamp': log_entry.get('__REALTIME_TIMESTAMP', ''),
                                'level': log_entry.get('PRIORITY', '')
                            })
                        except:
                            pass
            except:
                logs = []
            
            result[key] = {
                'name': info['name'],
                'description': info['description'],
                'service': service_name,
                'active': active,
                'state': service_info.get('ActiveState', 'unknown'),
                'substate': service_info.get('SubState', 'unknown'),
                'pid': service_info.get('MainPID', ''),
                'loaded': service_info.get('LoadState', 'unknown') == 'loaded',
                'recent_logs': logs[-3:] if logs else [],  # 最近3条日志
                'frontend_machine': {
                    'host': '192.168.10.131',
                    'description': '前置机 - MT5和连接器所在机器'
                } if key == 'mt5-collector' else None
            }
            
        except subprocess.TimeoutExpired:
            result[key] = {
                'name': info['name'],
                'description': info['description'],
                'service': service_name,
                'active': False,
                'error': '查询超时'
            }
        except Exception as e:
            logger.error(f"获取服务 {service_name} 状态失败: {e}")
            result[key] = {
                'name': info['name'],
                'description': info['description'],
                'service': service_name,
                'active': False,
                'error': str(e)
            }
    
    return jsonify({
        'success': True,
        'services': result
    })


@app.route('/api/system/service/<service_name>/control', methods=['POST'])
def control_service(service_name):
    """控制服务（启动/停止/重启）"""
    import subprocess
    
    action = request.json.get('action', 'status')  # start, stop, restart, status
    
    valid_services = ['mt5-collector', 'hft-core', 'backend-api']
    if service_name not in valid_services:
        return jsonify({
            'success': False,
            'error': f'无效的服务名称: {service_name}'
        }), 400
    
    service_file = f'{service_name}.service'
    
    try:
        if action == 'start':
            cmd = ['systemctl', 'start', service_file]
            action_name = '启动'
        elif action == 'stop':
            cmd = ['systemctl', 'stop', service_file]
            action_name = '停止'
        elif action == 'restart':
            cmd = ['systemctl', 'restart', service_file]
            action_name = '重启'
        else:
            return jsonify({
                'success': False,
                'error': f'无效的操作: {action}'
            }), 400
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            return jsonify({
                'success': True,
                'message': f'{action_name}服务成功'
            })
        else:
            return jsonify({
                'success': False,
                'error': result.stderr or f'{action_name}服务失败'
            }), 500
            
    except subprocess.TimeoutExpired:
        return jsonify({
            'success': False,
            'error': '操作超时'
        }), 500
    except Exception as e:
        logger.error(f"控制服务 {service_name} 失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/redis/status', methods=['GET'])
def get_redis_status():
    """获取Redis连接状态和数据统计"""
    try:
        # 检查Redis连接
        redis_client.ping()
        
        # 获取统计信息
        info = redis_client.info()
        
        # 统计TICK数据
        tick_keys = redis_client.keys("tick:*")
        history_keys = redis_client.keys("tick_history:*")
        
        status = {
            'connected': True,
            'redis_version': info.get('redis_version', 'unknown'),
            'used_memory': info.get('used_memory_human', 'unknown'),
            'connected_clients': info.get('connected_clients', 0),
            'tick_symbols': len(tick_keys),
            'history_symbols': len(history_keys),
            'uptime': info.get('uptime_in_seconds', 0)
        }
        
        return jsonify({
            'success': True,
            'status': status
        })
        
    except Exception as e:
        logger.error(f"获取Redis状态失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'status': {'connected': False}
        }), 500


@app.route('/api/system/connections', methods=['GET'])
def get_system_connections():
    """获取所有系统连接状态（MT5中继器、gRPC、Redis、后端）"""
    import requests
    
    connections = {
        'backend': {'connected': True, 'status': 'ok'},  # 后端本身总是连接的
        'redis': {'connected': False, 'status': 'unknown'},
        'mt5_relay': {'connected': False, 'status': 'unknown'},
        'grpc': {'connected': False, 'status': 'unknown'},
    }
    
    # 1. 检查 Redis 连接（快速检查，避免阻塞）
    try:
        redis_client.ping()
        # 🔴 修复：移除info()调用，避免阻塞（info()可能在某些情况下慢）
        connections['redis'] = {
            'connected': True,
            'status': 'ok',
            'version': '7.0.15'  # 直接返回已知版本，避免info()阻塞
        }
    except Exception as e:
        connections['redis'] = {
            'connected': False,
            'status': 'error',
            'error': str(e)[:50]  # 截断错误信息，避免过长
        }
    
    # 2. 检查 gRPC 服务连接（Windows 主机上的中继服务）
    # 🔴 架构说明（根据docs/系统架构/后端架构.md）：
    # - Windows中继服务 = gRPC服务（50051）+ ZeroMQ服务（5555）
    # - gRPC服务：处理查询和指令（同步），包括订单执行
    # - ZeroMQ服务：Windows端内部通信（MQL EA → Python），不用于订单执行
    # - 订单执行走gRPC，ZeroMQ只用于Windows端内部事件推送
    # 🔴 修复：使用超时保护，避免阻塞API响应
    from src.trading.services.grpc_trade_client import get_grpc_client, is_grpc_available
    grpc_host = '192.168.10.131'
    grpc_port = 50051
    grpc_address = f"{grpc_host}:{grpc_port}"
    
    # 初始化gRPC和MT5中继器状态
    grpc_connected = False
    mt5_relay_connected = False
    grpc_error = None
    mt5_relay_error = None
    
    # 🔴 修复：使用ThreadPoolExecutor包装gRPC检查，设置严格超时，避免阻塞
    # 如果gRPC检查超时，直接标记为未连接
    try:
        import concurrent.futures
        # 使用线程池执行器，设置0.5秒超时（健康检查必须快速）
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(is_grpc_available)
            try:
                grpc_available = future.result(timeout=0.5)  # 0.5秒超时
                if grpc_available:
                    # gRPC功能可用，但不进行实际连接检查（避免网络延迟）
                    grpc_connected = False
                    mt5_relay_connected = False
                    grpc_error = "gRPC功能可用但未连接（跳过连接检查以避免阻塞）"
                else:
                    grpc_connected = False
                    mt5_relay_connected = False
                    grpc_error = 'gRPC功能不可用'
            except concurrent.futures.TimeoutError:
                # 超时：直接标记为未连接
                grpc_connected = False
                mt5_relay_connected = False
                grpc_error = 'gRPC检查超时（0.5秒）'
    except Exception as e:
        grpc_connected = False
        mt5_relay_connected = False
        grpc_error = f'gRPC检查异常: {str(e)[:50]}'
    
    # 设置gRPC连接状态
    connections['grpc'] = {
        'connected': grpc_connected,
        'status': 'ok' if grpc_connected else 'error',
        'address': grpc_address,
        'error': grpc_error if not grpc_connected else None
    }
    
    # 设置MT5中继器连接状态（与gRPC状态一致）
    connections['mt5_relay'] = {
        'connected': mt5_relay_connected,
        'status': 'ok' if mt5_relay_connected else 'error',
        'mt5_connected': mt5_relay_connected,
        'service_status': 'gRPC服务运行中' if mt5_relay_connected else 'gRPC服务未连接',
        'address': grpc_address,
        'protocol': 'gRPC',
        'error': mt5_relay_error if not mt5_relay_connected else None,
        'hint': '请检查 Windows 主机上的 gRPC 服务是否运行 (mt5_relay_service.py)' if not mt5_relay_connected else None
    }
    
    return jsonify({
        'success': True,
        'connections': connections,
        'timestamp': time.time()
    })


@app.route('/api/system/clients', methods=['GET'])
def get_connected_clients():
    """获取当前连接的 WebSocket 客户端信息"""
    try:
        client_list = []
        for sid, client_info in clients.items():
            client_list.append({
                'sid': sid,
                'last_tick_time': client_info.get('last_time', 0),
                'connected_at': client_info.get('connected_at', None)
            })
        
        return jsonify({
            'success': True,
            'clients': client_list,
            'total_count': len(clients),
            'timestamp': time.time()
        })
    except Exception as e:
        logger.error(f"获取客户端信息失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'clients': [],
            'total_count': 0
        }), 500


# ==================== 标注API ====================

@app.route('/api/annotations/save', methods=['POST'])
def save_annotations():
    """保存标注数据到Redis"""
    try:
        data = request.json
        
        # 生成时间戳
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 1. 保存最新版本
        redis_client.set('annotations:hourly:latest', json.dumps(data, ensure_ascii=False))
        
        # 2. 保存历史版本（带时间戳）
        history_key = f'annotations:hourly:history:{timestamp}'
        redis_client.set(history_key, json.dumps(data, ensure_ascii=False))
        
        # 3. 更新统计信息
        stats = {
            'total_annotations': len(data.get('annotations', [])),
            'last_updated': timestamp,
            'data_quality': data.get('stats', {}).get('dataQuality', {})
        }
        redis_client.set('annotations:hourly:stats', json.dumps(stats, ensure_ascii=False))
        
        # 4. 将历史版本键加入列表（用于查询）
        redis_client.zadd('annotations:hourly:versions', {history_key: int(time.time())})
        
        logger.info(f"✅ 保存标注数据成功: {len(data.get('annotations', []))} 个标注")
        
        return jsonify({
            'success': True,
            'message': f'✅ 保存成功：{len(data.get("annotations", []))} 个标注',
            'timestamp': timestamp
        })
    
    except Exception as e:
        logger.error(f'保存标注失败: {str(e)}')
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/annotations/load', methods=['GET'])
def load_annotations():
    """从Redis加载最新标注数据"""
    try:
        # 获取最新数据
        data_str = redis_client.get('annotations:hourly:latest')
        
        if data_str:
            data = json.loads(data_str)
            return jsonify({
                'success': True,
                'data': data
            })
        else:
            return jsonify({
                'success': True,
                'data': None,
                'message': '暂无标注数据'
            })
    
    except Exception as e:
        logger.error(f'加载标注失败: {str(e)}')
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ==================== 启动 ====================

if __name__ == '__main__':
    # 配置日志
    logger.add(
        LOG_DIR / "gui_{time}.log",
        rotation="1 day",
        retention="30 days",
        level="INFO"
    )
    
    # 🔴 修复：移除启动时数据初始化，避免阻塞API Server启动
    # 数据初始化应该由Data Integrity Service在后台完成，而不是在API Server启动时阻塞
    # try:
    #     from src.trading.services.data_integrity_checker import initialize_data_on_startup
    #     logger.info("🔧 API Server启动：初始化历史数据...")
    #     initialize_data_on_startup(symbol="BTCUSDm", count=2880)
    # except Exception as e:
    #     logger.warning(f"启动时数据初始化失败（非关键）: {e}")
    
    logger.info("\n" + "="*60)
    logger.info("AI交易系统 - GUI服务启动")
    logger.info("="*60 + "\n")
    
    # 暂时不加载模型
    # logger.info("正在加载AI模型...")
    # best_model_path = CHECKPOINT_DIR / "best_model.pth"
    # load_model(str(best_model_path) if best_model_path.exists() else None)
    
    # 【Flask-SocketIO最佳实践】使用 socketio.start_background_task 而不是 threading.Thread
    # 参考：https://flask-socketio.readthedocs.io/en/latest/getting_started.html#background-tasks
    logger.info("启动实时数据广播后台任务...")
    socketio.start_background_task(broadcast_realtime_data)
    
    logger.info("启动持仓实时更新后台任务（gRPC）...")
    socketio.start_background_task(broadcast_positions_updates)
    
    logger.info("启动订单反馈监听后台任务...")
    socketio.start_background_task(listen_order_feedback)
    
    logger.info("启动 Redis Pub/Sub 订阅（事件驱动模式，接收 Windows gRPC 服务推送）...")
    socketio.start_background_task(listen_redis_pubsub)
    logger.info("  - 订阅频道: tick:*, kline:*, mt5:position_update, mt5:deal, mt5:trade_events 等")
    
    # 启动Flask服务
    logger.info("\n🚀 服务启动:")
    logger.info("  - API地址: http://localhost:5000")
    logger.info("  - WebSocket: ws://localhost:5000")
    logger.info("\n")
    
    # 使用 socketio.run() 而不是 app.run()
    # 注意：threaded 参数只在 threading 模式下有效，eventlet 模式不支持
    try:
        run_kwargs = {
            'app': app,
            'host': '0.0.0.0',
            'port': 5000,
            'debug': False,
            'use_reloader': False,
            'log_output': True,
            'allow_unsafe_werkzeug': True,
            # 🔴 注意：socketio.run() 不支持 backlog 参数，连接队列由底层socket控制
        }
        # 只在 threading 模式下添加 threaded 参数
        if async_mode == 'threading':
            run_kwargs['threaded'] = True
        
        socketio.run(**run_kwargs)
    except Exception as e:
        logger.error(f"Flask-SocketIO 启动失败: {e}", exc_info=True)
        raise
fo=True)
        raise
