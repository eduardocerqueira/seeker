#date: 2025-11-13T17:02:49Z
#url: https://api.github.com/gists/f99a8104b419a7b2f6437a18ce2a8790
#owner: https://api.github.com/users/wangwei334455

#!/usr/bin/env python3
"""
统一后端启动程序 - 一键启动所有后端组件

启动所有后端服务：
1. API Server (Flask + Socket.IO) - 前端接口和WebSocket
2. Data Puller (数据采集器) - 从MT5中继服务接收TICK数据
3. L2 Strategy Core (策略核心) - 策略决策和K线构建
4. OrderExecutor (订单执行器) - 执行交易指令

使用方式：
    python3 scripts/start_backend.py

或者作为模块运行：
    python3 -m scripts.start_backend
"""
import sys
import os
import signal
import time
import threading
import subprocess
from pathlib import Path
from loguru import logger

# 添加项目路径
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "src"))

# 配置日志
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger.add(
    LOG_DIR / "backend_main_{time}.log",
    rotation="10 MB",
    retention="7 days",
    level="INFO"
)

# 全局变量：存储所有服务线程
services = {}
stop_event = threading.Event()


def start_api_server():
    """启动API服务器"""
    logger.info("=" * 70)
    logger.info("启动服务 1/3: API Server (Flask + Socket.IO)")
    logger.info("=" * 70)
    
    try:
        # 直接运行api_server.py作为子进程（避免导入冲突）
        import subprocess
        import os
        
        def run_api_server():
            logger.info("🚀 API Server 启动中...")
            logger.info("  - API地址: http://localhost:5000")
            logger.info("  - WebSocket: ws://localhost:5000")
            
            # 切换到项目根目录
            os.chdir(BASE_DIR)
            
            # 运行api_server.py
            # 注意：不要使用 PIPE，否则会导致缓冲区阻塞
            # 🔴 修复：确保使用虚拟环境的 Python，并激活虚拟环境
            venv_python = BASE_DIR / "venv" / "bin" / "python3"
            python_executable = str(venv_python) if venv_python.exists() else sys.executable
            
            # 🔴 修复：设置环境变量，确保使用虚拟环境的Python和依赖
            env = dict(os.environ)
            env['PYTHONPATH'] = str(BASE_DIR)
            if venv_python.exists():
                # 如果使用虚拟环境，确保PATH包含虚拟环境的bin目录
                venv_bin = str(BASE_DIR / "venv" / "bin")
                env['PATH'] = f"{venv_bin}:{env.get('PATH', '')}"
            
            process = subprocess.Popen(
                [python_executable, str(BASE_DIR / "scripts" / "api_server.py")],
                stdout=subprocess.DEVNULL,  # 重定向到 /dev/null，避免缓冲区阻塞
                stderr=subprocess.PIPE,  # 保留 stderr 用于错误诊断
                cwd=str(BASE_DIR),
                bufsize=0,  # 无缓冲
                env=env  # 🔴 修复：使用完整的环境变量
            )
            services['api_server_process'] = process
            
            # 等待进程结束
            process.wait()
        
        thread = threading.Thread(target=run_api_server, daemon=True, name="API-Server")
        thread.start()
        services['api_server'] = thread
        
        # 等待API服务器启动
        import requests
        for i in range(30):
            try:
                response = requests.get("http://localhost:5000/api/health", timeout=1)
                if response.status_code == 200:
                    logger.info("✅ API Server 启动成功")
                    return True
            except:
                pass
            time.sleep(1)
        
        logger.error("❌ API Server 启动超时")
        return False
        
    except Exception as e:
        logger.error(f"❌ API Server 启动失败: {e}", exc_info=True)
        return False


def start_order_executor():
    """启动订单执行器"""
    logger.info("=" * 70)
    logger.info("启动服务 2/3: OrderExecutor (订单执行器)")
    logger.info("=" * 70)
    
    try:
        from src.trading.execution.order_executor import OrderExecutor
        
        def run_order_executor():
            logger.info("🚀 OrderExecutor 启动中...")
            logger.info("  - 监听队列: l3:manual:commands (Redis Stream)")
            logger.info("  - 监听队列: l2:order:commands (Redis List)")
            logger.info("  - 执行方式: 通过Windows MT5 gRPC服务")
            
            # 创建OrderExecutor实例（会自动启动监听线程）
            executor = OrderExecutor(symbol="BTCUSDm")
            services['order_executor_instance'] = executor
            
            logger.info("✅ OrderExecutor 监听线程已启动")
            
            # 保持运行直到停止信号
            while not stop_event.is_set():
                time.sleep(1)
            
            # 停止执行器
            if hasattr(executor, 'stop'):
                executor.stop()
            else:
                executor.stop_event.set()
            logger.info("OrderExecutor 已停止")
        
        thread = threading.Thread(target=run_order_executor, daemon=True, name="OrderExecutor")
        thread.start()
        services['order_executor'] = thread
        
        # 等待一下确保启动
        time.sleep(2)
        logger.info("✅ OrderExecutor 启动成功")
        return True
        
    except Exception as e:
        logger.error(f"❌ OrderExecutor 启动失败: {e}", exc_info=True)
        return False


def start_data_puller(optional: bool = False):
    """
    启动数据接收服务
    
    Data Puller功能：
    - 从Windows MT5中继服务被动接收TICK数据（gRPC StreamTicks，服务器推送）
    - 存储TICK数据到Redis（Sorted Set + Stream）
    - 为L2策略核心提供实时TICK数据流
    - 是系统数据流的起点，必须运行
    
    工作模式：
    - 客户端建立gRPC连接（主动）
    - 服务器持续推送TICK数据流（被动接收）
    """
    logger.info("=" * 70)
    logger.info("启动服务 2/4: Data Puller (数据采集器)")
    logger.info("=" * 70)
    
    try:
        import subprocess
        import os
        
        def run_data_puller():
            logger.info("🚀 Data Puller 启动中...")
            logger.info("  - 数据源: Windows MT5中继服务 (gRPC StreamTicks)")
            logger.info("  - 工作模式: 被动接收服务器推送的TICK数据流")
            logger.info("  - 存储: Redis (tick:BTCUSDm)")
            logger.info("  - 用途: 为L2策略核心提供实时TICK数据")
            
            # 切换到项目根目录
            os.chdir(BASE_DIR)
            
            # 运行data_puller.py
            # 🔴 修复：确保使用虚拟环境的 Python，并激活虚拟环境
            venv_python = BASE_DIR / "venv" / "bin" / "python3"
            python_executable = str(venv_python) if venv_python.exists() else sys.executable
            
            # 🔴 修复：设置环境变量，确保使用虚拟环境的Python和依赖
            env = dict(os.environ)
            env['PYTHONPATH'] = str(BASE_DIR)
            if venv_python.exists():
                # 如果使用虚拟环境，确保PATH包含虚拟环境的bin目录
                venv_bin = str(BASE_DIR / "venv" / "bin")
                env['PATH'] = f"{venv_bin}:{env.get('PATH', '')}"
            
            process = subprocess.Popen(
                [python_executable, str(BASE_DIR / "scripts" / "data_puller.py")],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(BASE_DIR),
                env=env  # 🔴 修复：使用完整的环境变量
            )
            services['data_puller_process'] = process
            
            # 等待进程结束
            process.wait()
        
        thread = threading.Thread(target=run_data_puller, daemon=True, name="DataPuller")
        thread.start()
        services['data_puller'] = thread
        
        # 等待一下确保启动
        time.sleep(2)
        logger.info("✅ Data Puller 启动成功")
        return True
        
    except Exception as e:
        logger.error(f"❌ Data Puller 启动失败: {e}", exc_info=True)
        return False


def start_data_integrity_service(optional: bool = False):
    """
    启动数据完整性检查服务
    
    数据完整性服务功能：
    - 验证TICK数据（seq检查、checksum验证）
    - 去重K线数据（相同时间戳只保留最新）
    - 补空K线数据（填充缺失的时间段）
    - 数据质量监控
    
    数据流：
    - 输入: Redis Stream (tick:BTCUSDm:stream) - 原始TICK流
    - 输出: Redis Stream (tick:BTCUSDm:validated:stream) - 验证后的TICK流
    - 输出: Redis Sorted Set (kline:BTCUSDm:1m) - 修复后的K线数据
    """
    logger.info("=" * 70)
    logger.info("启动服务 3/5: Data Integrity Service (数据完整性检查)")
    logger.info("=" * 70)
    
    try:
        from src.trading.services.data_integrity_service import DataIntegrityService
        
        def run_integrity_service():
            logger.info("🚀 数据完整性检查服务启动中...")
            logger.info("  - 功能: TICK验证、K线去重、K线补空")
            logger.info("  - 输入: Redis Stream (tick:BTCUSDm:stream)")
            logger.info("  - 输出: Redis Stream (tick:BTCUSDm:validated:stream)")
            
            # 创建数据完整性服务实例
            integrity_service = DataIntegrityService(symbol="BTCUSDm")
            services['integrity_service_instance'] = integrity_service
            
            # 启动服务
            integrity_service.start()
            
            # 保持运行直到停止信号
            while not stop_event.is_set():
                time.sleep(1)
            
            # 停止服务
            integrity_service.stop()
            logger.info("数据完整性检查服务已停止")
        
        thread = threading.Thread(target=run_integrity_service, daemon=True, name="DataIntegrityService")
        thread.start()
        services['integrity_service'] = thread
        
        # 等待一下确保启动
        time.sleep(2)
        logger.info("✅ 数据完整性检查服务启动成功")
        return True
        
    except Exception as e:
        logger.error(f"❌ 数据完整性检查服务启动失败: {e}", exc_info=True)
        return False


def start_kline_service(optional: bool = False):
    """
    启动K线服务（独立的K线构建服务）
    
    Kline Service功能：
    - 监听已验证的TICK流，构建K线
    - 存储历史K线到Redis
    - 推送当前K线到Redis Pub/Sub（供前端实时显示）
    
    数据流：
    - 输入: Redis Stream (tick:BTCUSDm:validated:stream) - 已验证的TICK流
    - 输出: Redis Sorted Set (kline:BTCUSDm:1m) - 历史K线
    - 输出: Redis Pub/Sub (current_kline:BTCUSDm:m1) - 当前K线（实时跳动）
    """
    logger.info("=" * 70)
    logger.info("启动服务 3/6: Kline Service (K线构建服务)")
    logger.info("=" * 70)
    
    try:
        from src.trading.services.kline_service import KlineService
        
        def run_kline_service():
            logger.info("🚀 K线服务启动中...")
            logger.info("  - 数据源: Redis Stream (tick:BTCUSDm:validated:stream)")
            logger.info("  - 功能: K线构建、存储、推送")
            logger.info("  - 输出: Redis (kline:BTCUSDm:1m, current_kline:BTCUSDm:m1)")
            
            # 创建K线服务实例
            kline_service = KlineService(symbol="BTCUSDm")
            services['kline_service_instance'] = kline_service
            
            # 启动服务
            kline_service.start()
            
            # 保持运行直到停止信号
            while not stop_event.is_set():
                time.sleep(1)
            
            # 停止服务
            kline_service.stop()
            logger.info("K线服务已停止")
        
        thread = threading.Thread(target=run_kline_service, daemon=True, name="KlineService")
        thread.start()
        services['kline_service'] = thread
        
        # 等待一下确保启动
        time.sleep(2)
        logger.info("✅ K线服务启动成功")
        return True
        
    except Exception as e:
        logger.error(f"❌ K线服务启动失败: {e}", exc_info=True)
        return False


def start_l2_strategy_core(optional: bool = False):
    """
    启动L2策略核心服务
    
    L2 Strategy Core功能：
    - 从已验证的Redis Stream消费TICK数据（不再负责数据验证）
    - 构建多周期K线（M1, M5, H1等）
    - 执行策略决策（FSM状态机）
    - 生成交易信号
    - 实时推送当前K线到Redis（供前端显示）
    
    数据流：
    - 输入: Redis Stream (tick:BTCUSDm:validated:stream) - 已验证的TICK流
    - 输出: Redis Stream (signal:BTCUSDm:stream)
    - 输出: Redis Pub/Sub (current_kline:BTCUSDm:1m) - 实时K线跳动
    """
    logger.info("=" * 70)
    logger.info("启动服务 4/5: L2 Strategy Core (策略核心)")
    logger.info("=" * 70)
    
    # 🚀 行业最佳实践：启动时初始化历史数据
    try:
        from src.trading.services.data_integrity_checker import initialize_data_on_startup
        logger.info("🔧 启动时数据初始化：从MT5拉取历史数据并补空...")
        initialize_data_on_startup(symbol="BTCUSDm", count=2880)
    except Exception as e:
        logger.warning(f"启动时数据初始化失败（非关键）: {e}")
    
    try:
        from src.trading.core.strategy_fsm import L2StrategyCore
        
        def run_l2_core():
            logger.info("🚀 L2 Strategy Core 启动中...")
            logger.info("  - 数据源: Redis Stream (tick:BTCUSDm:validated:stream) - 已验证的TICK流")
            logger.info("  - 功能: K线构建、策略决策、信号生成")
            logger.info("  - 输出: Redis Stream (signal:BTCUSDm:stream)")
            logger.info("  - 实时推送: current_kline:BTCUSDm:1m (K线跳动)")
            
            # 创建L2核心实例
            l2_core = L2StrategyCore(symbol="BTCUSDm")
            services['l2_core_instance'] = l2_core
            
            logger.info("✅ L2 Strategy Core 初始化成功")
            
            # 保持运行直到停止信号
            while not stop_event.is_set():
                time.sleep(1)
            
            # 停止L2核心
            l2_core.stop()
            logger.info("L2 Strategy Core 已停止")
        
        thread = threading.Thread(target=run_l2_core, daemon=True, name="L2StrategyCore")
        thread.start()
        services['l2_core'] = thread
        
        # 等待一下确保启动
        time.sleep(2)
        logger.info("✅ L2 Strategy Core 启动成功")
        return True
        
    except Exception as e:
        logger.error(f"❌ L2 Strategy Core 启动失败: {e}", exc_info=True)
        return False


def signal_handler(sig, frame):
    """信号处理函数"""
    logger.info("")
    logger.info("收到停止信号，正在优雅关闭所有服务...")
    stop_event.set()
    
    # 停止所有子进程
    for name in ['api_server_process', 'data_puller_process']:
        if name in services:
            process = services[name]
            if isinstance(process, subprocess.Popen) and process.poll() is None:
                logger.info(f"停止 {name} 进程...")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except:
                    process.kill()
    
    # 停止OrderExecutor实例
    if 'order_executor_instance' in services:
        executor = services['order_executor_instance']
        if hasattr(executor, 'stop'):
            executor.stop()
        elif hasattr(executor, 'stop_event'):
            executor.stop_event.set()
    
    # 停止数据完整性服务实例
    if 'integrity_service_instance' in services:
        integrity_service = services['integrity_service_instance']
        if hasattr(integrity_service, 'stop'):
            integrity_service.stop()
        logger.info("数据完整性检查服务已停止")
    
    # 停止K线服务实例
    if 'kline_service_instance' in services:
        kline_service = services['kline_service_instance']
        if hasattr(kline_service, 'stop'):
            kline_service.stop()
        logger.info("K线服务已停止")
    
    # 停止L2 Strategy Core实例
    if 'l2_core_instance' in services:
        l2_core = services['l2_core_instance']
        if hasattr(l2_core, 'stop'):
            l2_core.stop()
        logger.info("L2 Strategy Core 已停止")
    
    # 等待所有线程结束
    for name, thread in services.items():
        if isinstance(thread, threading.Thread) and thread.is_alive():
            logger.info(f"等待 {name} 停止...")
            thread.join(timeout=5)
    
    logger.info("所有服务已停止")
    sys.exit(0)


def main():
    """主函数"""
    logger.info("\n" + "=" * 70)
    logger.info("AI交易系统 - 统一后端启动程序")
    logger.info("=" * 70)
    logger.info("")
    logger.info("将启动以下服务：")
    logger.info("  1. API Server (Flask + Socket.IO) - 端口 5000")
    logger.info("  2. Data Puller (数据采集器) - 从MT5中继服务接收TICK数据")
    logger.info("  3. Data Integrity Service (数据完整性检查) - TICK验证、K线去重补空")
    logger.info("  4. Kline Service (K线构建服务) - 构建K线、存储、推送")
    logger.info("  5. L2 Strategy Core (策略核心) - 策略决策（从Redis读取K线）")
    logger.info("  6. OrderExecutor (订单执行器) - 监听Redis队列执行交易")
    logger.info("")
    logger.info("💡 数据流说明：")
    logger.info("  Windows MT5中继 → Data Puller (gRPC StreamTicks)")
    logger.info("  → Redis Stream (原始) → Data Integrity Service (验证、去重、补空)")
    logger.info("  → Redis Stream (已验证) → Kline Service (构建K线)")
    logger.info("  → Redis Stream (已验证) → L2 Strategy Core (策略决策，从Redis读取K线)")
    logger.info("  → 策略决策 → OrderExecutor → Windows MT5中继")
    logger.info("  → API Server (WebSocket) → 前端显示")
    logger.info("")
    
    # 注册信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 启动服务（按依赖顺序）
    results = []
    
    # 1. 启动API服务器（前端接口）
    results.append(("API Server", start_api_server()))
    time.sleep(2)
    
    # 2. 启动数据采集器（必须最先运行，提供TICK数据流）
    results.append(("Data Puller", start_data_puller(optional=False)))
    time.sleep(2)
    
    # 3. 启动数据完整性检查服务（验证、去重、补空）
    results.append(("Data Integrity Service", start_data_integrity_service(optional=False)))
    time.sleep(2)
    
    # 4. 启动K线服务（构建K线，供前端和策略服务使用）
    results.append(("Kline Service", start_kline_service(optional=False)))
    time.sleep(2)
    
    # 5. 启动L2策略核心（依赖已验证的TICK数据流和K线数据）
    results.append(("L2 Strategy Core", start_l2_strategy_core(optional=False)))
    time.sleep(2)
    
    # 6. 启动订单执行器（依赖L2核心生成信号）
    results.append(("OrderExecutor", start_order_executor()))
    time.sleep(1)
    
    # 检查启动结果
    logger.info("")
    logger.info("=" * 70)
    logger.info("启动结果")
    logger.info("=" * 70)
    
    for name, success in results:
        status = "✅ 成功" if success else "❌ 失败"
        logger.info(f"  {name}: {status}")
    
    success_count = sum(1 for _, success in results if success)
    total_count = len(results)
    
    if success_count < total_count:
        logger.warning(f"⚠️ 部分服务启动失败 ({success_count}/{total_count})")
    else:
        logger.info("")
        logger.info("🎉 所有核心服务已启动！")
        logger.info("")
        logger.info("服务状态：")
        logger.info("  - API Server: http://localhost:5000")
        logger.info("  - Data Puller: 连接Windows MT5中继服务 (192.168.10.131:50051)")
        logger.info("  - Data Integrity Service: 验证TICK数据，检查K线完整性（去重、补空）")
        logger.info("  - L2 Strategy Core: 处理已验证的TICK数据，构建K线，执行策略")
        logger.info("  - OrderExecutor: 监听Redis队列，执行交易指令")
        logger.info("")
        logger.info("💡 确保Windows MT5中继服务已启动！")
        logger.info("   Windows端: python mt5_relay_service.py")
        logger.info("")
        logger.info("按 Ctrl+C 停止所有服务")
        logger.info("")
    
    # 保持主线程运行
    try:
        while True:
            time.sleep(1)
            
            # 检查服务是否还在运行
            for name, item in list(services.items()):
                if isinstance(item, threading.Thread):
                    if not item.is_alive():
                        logger.warning(f"⚠️ 服务 {name} 已停止")
                        services.pop(name, None)
                elif isinstance(item, subprocess.Popen):
                    if item.poll() is not None:
                        logger.warning(f"⚠️ 服务 {name} 已停止 (退出码: {item.returncode})")
                        services.pop(name, None)
                    
    except KeyboardInterrupt:
        signal_handler(None, None)


if __name__ == "__main__":
    main()

